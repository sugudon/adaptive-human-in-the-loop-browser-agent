import os
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

API = os.getenv("API_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="AI Web Automation Agent", page_icon="🤖", layout="wide")

if "task_id" not in st.session_state: st.session_state.task_id = None
if "task" not in st.session_state: st.session_state.task = None

ACTIVE_STATUSES = {"OBSERVING", "THINKING", "EXECUTING", "RECOVERING"}


def api(method, endpoint, timeout=120, **kwargs):
    try:
        r = requests.request(method, f"{API}{endpoint}", timeout=timeout, **kwargs)
        if not r.ok: st.error(r.text)
        return r
    except requests.RequestException as exc:
        st.error(f"Backend connection error: {exc}")
        return None


def refresh():
    if st.session_state.task_id:
        r = api("GET", f"/tasks/{st.session_state.task_id}", timeout=20)
        if r and r.ok: st.session_state.task = r.json()


refresh()

# Poll every 2s while the agent is actively running so the UI reflects the
# autonomous background loop without the user clicking Refresh.
if st.session_state.task and st.session_state.task.get("status") in ACTIVE_STATUSES:
    st_autorefresh(interval=2000, key="agent_poll")

st.title("🤖 AI Web Automation Agent")
st.caption("Autonomous Observe → Decide → Execute with Human Intervention")

with st.sidebar:
    st.header("Agent")
    st.write("LLM: Ollama / Gemma 2B")
    st.write("Browser: Playwright")
    st.write("Mode: Autonomous + Human Takeover")
    if st.button("🔄 Refresh", use_container_width=True): refresh(); st.rerun()
    if st.button("🗑️ Clear Task", use_container_width=True):
        st.session_state.task_id = None; st.session_state.task = None; st.rerun()

prompt = st.text_area("What should the browser do?", placeholder="Open a website and complete the task...", height=120)
if st.button("🚀 Start Agent", type="primary", use_container_width=True):
    if not prompt.strip(): st.warning("Enter a task first.")
    else:
        r = api("POST", "/tasks", json={"task": prompt.strip()}, timeout=20)
        if r and r.ok:
            task = r.json(); st.session_state.task_id = task["id"]; st.session_state.task = task
            r2 = api("POST", f"/tasks/{task['id']}/start", timeout=30)
            if r2 and r2.ok: st.session_state.task = r2.json()
            elif r2 is not None and r2.status_code == 409:
                st.warning("Another task is already using the browser. Pause it first.")
            st.rerun()

task = st.session_state.task
if task:
    status = task.get("status", "UNKNOWN")
    icons = {"CREATED":"🟡","OBSERVING":"👀","THINKING":"🧠","EXECUTING":"▶️","RECOVERING":"🔄","WAITING_FOR_HUMAN":"👤","PAUSED":"⏸️","COMPLETED":"✅","FAILED":"❌"}
    st.divider(); st.subheader("Current Task"); st.info(task.get("user_prompt", ""))
    st.metric("Agent Status", f"{icons.get(status,'⚪')} {status}")

    if status == "WAITING_FOR_HUMAN":
        intervention = task.get("intervention") or {}
        st.error(f"👤 Human intervention required: {intervention.get('type','UNKNOWN')}")
        st.write(intervention.get("reason", "The browser requires human attention."))
        st.info("Complete the verification directly in the opened browser. Do not close the browser.")
        if st.button("▶️ Resume Agent", type="primary", use_container_width=True):
            r = api("POST", f"/tasks/{task['id']}/intervention/resume", timeout=30)
            if r and r.ok: st.session_state.task = r.json(); st.rerun()

    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("⏸️ Pause", use_container_width=True):
            r=api("POST",f"/tasks/{task['id']}/pause",timeout=30)
            if r and r.ok: st.session_state.task=r.json(); st.rerun()
    with c2:
        if st.button("▶️ Resume", use_container_width=True):
            r=api("POST",f"/tasks/{task['id']}/resume",timeout=30)
            if r and r.ok: st.session_state.task=r.json(); st.rerun()
            elif r is not None and r.status_code == 409:
                st.warning("Another task is already using the browser. Pause it first.")
    with c3:
        if st.button("👀 Observe Now", use_container_width=True):
            r=api("POST",f"/tasks/{task['id']}/observe",timeout=60)
            if r and r.ok: st.session_state.task=r.json()["task"]; st.rerun()

    intervention = task.get("intervention")
    if intervention: st.warning(f"⚠️ {intervention.get('type')}: {intervention.get('reason')}")

    steps = task.get("steps", [])
    if steps:
        st.subheader("🤖 Agent Actions")
        for step in steps[-10:]:
            with st.expander(f"Step {step.get('id')} — {step.get('action')} — {step.get('status')}"):
                st.write(f"Target: `{step.get('target','')}`")
                st.write(f"Value: `{step.get('value','')}`")
                if step.get("reason"): st.write(f"Reason: {step['reason']}")
                if step.get("status") in {"FAILED", "READY"}:
                    new_target=st.text_input("Target",step.get("target",""),key=f"t{step['id']}")
                    new_value=st.text_input("Value",step.get("value",""),key=f"v{step['id']}")
                    if st.button("💾 Update & Retry",key=f"u{step['id']}"):
                        r=api("PUT",f"/tasks/{task['id']}/steps/{step['id']}",json={"target":new_target,"value":new_value},timeout=20)
                        if r and r.ok:
                            r2=api("POST",f"/tasks/{task['id']}/replay/{step['id']}",timeout=60)
                            if r2 and r2.ok: st.session_state.task=r2.json()
                        st.rerun()
                    if step.get("status")=="FAILED" and st.button("🧠 Replan",key=f"r{step['id']}"):
                        r=api("POST",f"/tasks/{task['id']}/replan",timeout=180)
                        if r and r.ok: st.session_state.task=r.json(); st.rerun()

    if task.get("observation"):
        st.subheader("👀 Latest Observation")
        with st.expander("Browser state", expanded=False): st.text(task["observation"])
    if task.get("last_error"): st.error(task["last_error"])
    if task.get("history"):
        with st.expander("📜 History"): st.json(task["history"][-20:])
