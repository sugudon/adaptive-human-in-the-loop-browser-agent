import asyncio
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent.agent import decide_next_action, replan, target_is_grounded, is_noop_navigate
from agent.state import Task, Step
from automation.executor import BrowserExecutor

app = FastAPI(title="AI Web Automation Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Streamlit origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

tasks: dict[str, Task] = {}
executor = BrowserExecutor()
runner_tasks: dict[str, asyncio.Task] = {}

# One shared BrowserExecutor drives one browser/page. Only one task may be
# actively OBSERVING/THINKING/EXECUTING against it at a time, or two tasks
# would silently interleave actions on the same page.
active_task_id: Optional[str] = None
ACTIVE_STATUSES = {"OBSERVING", "THINKING", "EXECUTING", "RECOVERING", "WAITING_FOR_HUMAN"}


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


class StepUpdate(BaseModel):
    value: Optional[str] = None
    target: Optional[str] = None
    reason: Optional[str] = None


def get_task(task_id: str) -> Task:
    if task_id not in tasks:
        raise HTTPException(404, "Task not found.")
    return tasks[task_id]


def claim_executor(task_id: str):
    global active_task_id
    if active_task_id is not None and active_task_id != task_id:
        other = tasks.get(active_task_id)
        if other and other.status in ACTIVE_STATUSES:
            raise HTTPException(
                409,
                f"Task {active_task_id} is currently using the browser. "
                f"Pause it before starting another task.",
            )
    active_task_id = task_id


def release_executor(task_id: str):
    global active_task_id
    if active_task_id == task_id:
        active_task_id = None


async def save_observation(task: Task):
    obs = await executor.get_page_snapshot()
    task.observation = json.dumps(obs, indent=2, ensure_ascii=False)
    url = obs.get("url")
    if url and url not in task.visited_urls:
        task.visited_urls.append(url)
    return obs


async def check_intervention(task: Task, observation=None) -> bool:
    result = await executor.detect_intervention()
    if result.get("required"):
        task.intervention = result
        task.status = "WAITING_FOR_HUMAN"
        task.history.append({"type": "INTERVENTION", **result})
        return True
    task.intervention = None
    return False


STUCK_LOOP_THRESHOLD = 3  # identical consecutive (action, target, value) triggers a pause


def _is_repeat_of_recent(step: Step, task: Task, n: int = STUCK_LOOP_THRESHOLD) -> bool:
    """True if the last n executed steps (excluding this one) are all the
    exact same (action, target, value) as this step — i.e. the agent is
    stuck looping without making progress."""
    fingerprint = (step.action, step.target, step.value)
    recent = [s for s in task.steps if s.id != step.id][-n:]
    if len(recent) < n:
        return False
    return all((s.action, s.target, s.value) == fingerprint for s in recent)


async def decide_and_append(task: Task, observation):
    action = await asyncio.to_thread(decide_next_action, task.user_prompt, observation, task.history, task.visited_urls)
    if action["action"] == "done":
        task.status = "COMPLETED"
        return None
    step = Step(id=len(task.steps) + 1, **action, status="READY")
    task.steps.append(step)
    task.status = "READY"
    return step


async def autonomous_loop(task: Task):
    """Observe -> Decide -> Execute continuously until completion/intervention/pause."""
    try:
        while not task.stop_requested and task.status not in {"COMPLETED", "FAILED", "WAITING_FOR_HUMAN", "PAUSED"}:
            task.status = "OBSERVING"
            observation = await save_observation(task)
            if await check_intervention(task, observation):
                return

            task.status = "THINKING"
            try:
                step = await decide_and_append(task, observation)
            except Exception as exc:
                task.status = "FAILED"
                task.last_error = f"LLM decision failed: {exc}"
                return
            if step is None:
                return

            noop_nav = is_noop_navigate(step.model_dump(), observation)
            ungrounded = not target_is_grounded(step.model_dump(), observation)

            if not (ungrounded or noop_nav) and _is_repeat_of_recent(step, task):
                task.intervention = {
                    "required": True,
                    "type": "STUCK_LOOP",
                    "reason": f"Agent repeated the same action {STUCK_LOOP_THRESHOLD}x in a row "
                              f"({step.action} '{step.target}') with no progress.",
                }
                task.status = "WAITING_FOR_HUMAN"
                task.history.append({"type": "INTERVENTION", **task.intervention})
                return

            task.current_step = len(task.steps) - 1
            step.status = "RUNNING"
            task.status = "EXECUTING"

            if ungrounded or noop_nav:
                # Either a selector that isn't on the real page (hallucination)
                # or a navigate to the URL we're already on (no-op). Don't
                # waste a Playwright call on it — go straight to replan with
                # that fact stated explicitly.
                step.status = "FAILED"
                if noop_nav:
                    guard_error = f"Already on this URL ('{observation.get('url')}') — navigating again is a no-op."
                else:
                    guard_error = f"Target '{step.target}' does not exist on the current page (not in snapshot)."
                task.last_error = guard_error
                task.history.append({"step_id": step.id, "action": step.action, "target": step.target, "value": step.value, "status": "FAILED", "error": guard_error})
                task.status = "RECOVERING"
                try:
                    new_action = await asyncio.to_thread(
                        replan, task.user_prompt, observation,
                        task.history, executor.page.url if executor.page else "",
                        {**step.model_dump(), "error": guard_error},
                        task.visited_urls,
                    )
                except Exception as replan_exc:
                    task.status = "FAILED"
                    task.last_error = f"Replan failed: {replan_exc}"
                    return

                replacement = Step(id=len(task.steps) + 1, **new_action, status="RUNNING")
                task.steps.append(replacement)
                if not target_is_grounded(replacement.model_dump(), observation) or is_noop_navigate(replacement.model_dump(), observation):
                    replacement.status = "FAILED"
                    reason = (
                        f"Replan produced another no-op navigate to '{replacement.target or replacement.value}'."
                        if is_noop_navigate(replacement.model_dump(), observation)
                        else f"Replan also produced an ungrounded target: '{replacement.target}'."
                    )
                    task.last_error = reason
                    task.intervention = {"required": True, "type": "STUCK_LOOP", "reason": reason}
                    task.status = "WAITING_FOR_HUMAN"
                    task.history.append({"type": "INTERVENTION", **task.intervention})
                    return
                try:
                    result = await executor.execute_step(replacement.action, replacement.target, replacement.value)
                    replacement.status = "SUCCESS"
                    replacement.result = result
                    task.history.append({"step_id": replacement.id, **new_action, "status": "REPLANNED_SUCCESS"})
                except Exception as retry_exc:
                    replacement.status = "FAILED"
                    task.last_error = str(retry_exc)
                    task.status = "FAILED"
                    return

                observation = await save_observation(task)
                if await check_intervention(task, observation):
                    return
                await asyncio.sleep(0.1)
                continue

            try:
                result = await executor.execute_step(step.action, step.target, step.value)
                step.status = "SUCCESS"
                step.result = result
                task.history.append({"step_id": step.id, "action": step.action, "target": step.target, "value": step.value, "status": "SUCCESS", "result": result})
            except Exception as exc:
                step.status = "FAILED"
                task.last_error = str(exc)
                task.history.append({"step_id": step.id, "action": step.action, "target": step.target, "value": step.value, "status": "FAILED", "error": str(exc)})
                task.status = "RECOVERING"
                try:
                    new_action = await asyncio.to_thread(
                        replan, task.user_prompt, await executor.get_page_snapshot(),
                        task.history, executor.page.url if executor.page else "", step.model_dump(),
                        task.visited_urls,
                    )
                except Exception as replan_exc:
                    task.status = "FAILED"
                    task.last_error = f"Replan failed: {replan_exc}"
                    return

                replacement = Step(id=len(task.steps) + 1, **new_action, status="RUNNING")
                task.steps.append(replacement)
                try:
                    result = await executor.execute_step(replacement.action, replacement.target, replacement.value)
                    replacement.status = "SUCCESS"
                    replacement.result = result
                    task.history.append({"step_id": replacement.id, **new_action, "status": "REPLANNED_SUCCESS"})
                except Exception as retry_exc:
                    replacement.status = "FAILED"
                    task.last_error = str(retry_exc)
                    task.status = "FAILED"
                    return

            observation = await save_observation(task)
            if await check_intervention(task, observation):
                return
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        task.status = "PAUSED"
        raise
    except Exception as exc:
        task.status = "FAILED"
        task.last_error = str(exc)
    finally:
        release_executor(task.id)


@app.get("/")
def home():
    return {"message": "AI Web Automation Agent API is running", "version": "2.0.0"}


@app.post("/tasks")
def create_task(request: TaskRequest):
    task = Task.create(request.task)
    tasks[task.id] = task
    return task


@app.post("/tasks/{task_id}/start")
async def start_task(task_id: str):
    task = get_task(task_id)
    claim_executor(task_id)
    task.stop_requested = False
    if task_id in runner_tasks and not runner_tasks[task_id].done():
        return task
    await executor.ensure_browser()
    task.status = "OBSERVING"
    runner_tasks[task_id] = asyncio.create_task(autonomous_loop(task))
    return task


@app.post("/tasks/{task_id}/observe")
async def observe_task(task_id: str):
    task = get_task(task_id)
    obs = await save_observation(task)
    await check_intervention(task, obs)
    return {"task": task, "observation": obs}


@app.post("/tasks/{task_id}/decide")
async def decide_task(task_id: str):
    task = get_task(task_id)
    obs = await save_observation(task)
    if await check_intervention(task, obs):
        return task
    task.status = "THINKING"
    await decide_and_append(task, obs)
    return task


@app.post("/tasks/{task_id}/execute")
async def execute_task(task_id: str):
    task = get_task(task_id)
    if task.current_step >= len(task.steps):
        raise HTTPException(400, "No current action.")
    step = task.steps[task.current_step]
    if step.status not in {"READY", "WAITING_APPROVAL"}:
        raise HTTPException(400, f"Step is not executable: {step.status}")
    step.status = "RUNNING"
    task.status = "EXECUTING"
    try:
        result = await executor.execute_step(step.action, step.target, step.value)
        step.status = "SUCCESS"
        step.result = result
        task.history.append({"step_id": step.id, "status": "SUCCESS", "result": result})
        obs = await save_observation(task)
        if await check_intervention(task, obs):
            return task
        task.current_step += 1
        task.status = "READY"
        return task
    except Exception as exc:
        step.status = "FAILED"
        task.last_error = str(exc)
        task.status = "FAILED"
        return task


@app.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    task = get_task(task_id)
    task.stop_requested = True
    runner = runner_tasks.get(task_id)
    if runner and not runner.done():
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass
    task.status = "PAUSED"
    release_executor(task_id)
    return task


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    task = get_task(task_id)
    claim_executor(task_id)
    task.stop_requested = False
    task.intervention = None
    task.last_error = None
    if task_id not in runner_tasks or runner_tasks[task_id].done():
        runner_tasks[task_id] = asyncio.create_task(autonomous_loop(task))
    task.status = "OBSERVING"
    return task


@app.post("/tasks/{task_id}/intervention/resume")
async def resume_after_human(task_id: str):
    task = get_task(task_id)
    claim_executor(task_id)
    task.intervention = None
    task.stop_requested = False
    task.history.append({"type": "HUMAN_RESUMED_AGENT"})
    if task_id not in runner_tasks or runner_tasks[task_id].done():
        runner_tasks[task_id] = asyncio.create_task(autonomous_loop(task))
    task.status = "OBSERVING"
    return task


@app.post("/tasks/{task_id}/skip")
def skip_step(task_id: str):
    task = get_task(task_id)
    if task.current_step < len(task.steps):
        step = task.steps[task.current_step]
        step.status = "SKIPPED"
        task.history.append({"step_id": step.id, "status": "SKIPPED"})
        task.current_step += 1
    task.status = "OBSERVING"
    return task


@app.put("/tasks/{task_id}/steps/{step_id}")
def update_step(task_id: str, step_id: int, update: StepUpdate):
    task = get_task(task_id)
    step = next((s for s in task.steps if s.id == step_id), None)
    if not step:
        raise HTTPException(404, "Step not found.")
    if update.value is not None:
        step.value = update.value
    if update.target is not None:
        step.target = update.target
    if update.reason is not None:
        step.reason = update.reason
    return task


@app.post("/tasks/{task_id}/replay/{step_id}")
async def replay_step(task_id: str, step_id: int):
    """Actually execute a step with whatever target/value/action it currently
    holds (typically just human-corrected via PUT /steps/{id}), then resume
    the autonomous loop so the agent continues from here — instead of the
    old behavior of only recording a REPLAYED history entry with no status
    change and no continuation."""
    task = get_task(task_id)
    step = next((s for s in task.steps if s.id == step_id), None)
    if not step:
        raise HTTPException(404, "Step not found.")

    claim_executor(task_id)
    step.status = "RUNNING"
    task.status = "EXECUTING"
    try:
        result = await executor.execute_step(step.action, step.target, step.value)
        step.status = "SUCCESS"
        step.result = result
        task.history.append({"step_id": step.id, "action": step.action, "target": step.target, "value": step.value, "status": "HUMAN_RETRY_SUCCESS", "result": result})
        task.last_error = None
        task.intervention = None
        task.current_step = next(i for i, s in enumerate(task.steps) if s.id == step.id) + 1

        obs = await save_observation(task)
        if await check_intervention(task, obs):
            return task

        # Keep going: resume the autonomous loop from this corrected point
        # rather than leaving the task idle after one manual fix.
        task.stop_requested = False
        if task_id not in runner_tasks or runner_tasks[task_id].done():
            runner_tasks[task_id] = asyncio.create_task(autonomous_loop(task))
        task.status = "OBSERVING"
        return task
    except Exception as exc:
        step.status = "FAILED"
        error_msg = str(exc)
        task.last_error = error_msg
        task.history.append({"step_id": step.id, "action": step.action, "target": step.target, "value": step.value, "status": "FAILED", "error": error_msg})
        task.intervention = {
            "required": True,
            "type": "RETRY_FAILED",
            "reason": f"Manual retry of step {step.id} also failed: {error_msg}",
        }
        task.status = "WAITING_FOR_HUMAN"
        release_executor(task_id)
        return task


@app.post("/tasks/{task_id}/replan")
async def replan_task(task_id: str):
    task = get_task(task_id)
    failed = next((s for s in reversed(task.steps) if s.status == "FAILED"), None)
    if not failed:
        raise HTTPException(400, "No failed step available for replanning.")
    observation = await save_observation(task)
    action = await asyncio.to_thread(
        replan, task.user_prompt, observation, task.history,
        executor.page.url if executor.page else "", failed.model_dump(),
        task.visited_urls,
    )
    step = Step(id=len(task.steps) + 1, **action, status="READY")
    task.steps.append(step)
    task.current_step = len(task.steps) - 1
    task.status = "READY"
    task.last_error = None
    return task


@app.get("/tasks/{task_id}")
def get_task_endpoint(task_id: str):
    return get_task(task_id)


@app.on_event("shutdown")
async def shutdown():
    for runner in runner_tasks.values():
        if not runner.done():
            runner.cancel()
    await executor.close()
