import json
import os
import re
import time
import requests # type: ignore

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_RETRIES = int(os.getenv("OLLAMA_RETRIES", "2"))

# This set must stay in sync with automation/executor.py's execute_step().
# Every action listed here MUST have a matching branch in the executor,
# or the agent will pick actions it can never actually run.
SUPPORTED_ACTIONS = {
    "navigate", "click", "fill", "press", "wait",
    "back", "scroll", "select", "extract_text", "done",
}

# Ollama structured-output schema: the "format" field below constrains the
# decoder itself (via grammar-based sampling) so the model CANNOT emit
# malformed JSON, an unknown field, or an action name outside
# SUPPORTED_ACTIONS. This is stronger than the old format:"json", which only
# asked for "some valid JSON" and left the model free to invent fields or
# action names — this makes those specific failure modes structurally
# impossible regardless of model size. It does not fix content mistakes
# (e.g. picking the wrong index) — only shape/vocabulary.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": sorted(SUPPORTED_ACTIONS)},
        "target": {"type": "string"},
        "value": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["action", "target", "value", "reason"],
}


def ask_gemma(prompt: str, format_schema=ACTION_SCHEMA) -> str:
    last_exc = None
    current_format = format_schema
    for attempt in range(1, OLLAMA_RETRIES + 2):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": current_format,
                    "options": {"temperature": 0.1},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.HTTPError as exc:
            # Older Ollama servers reject a schema object for "format" (they
            # only understand the plain "json" string). Fall back once and
            # keep going instead of hard-failing the whole task over a
            # version mismatch.
            if isinstance(current_format, dict) and exc.response is not None and exc.response.status_code == 400:
                print("⚠️ Ollama rejected schema-based format (older server?) — falling back to format:'json'.")
                current_format = "json"
                continue
            last_exc = exc
            break
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt <= OLLAMA_RETRIES:
                time.sleep(1.5 * attempt)
                continue
        except requests.RequestException as exc:
            last_exc = exc
            break
    raise RuntimeError(
        f"Could not reach Ollama at {OLLAMA_URL} (model={OLLAMA_MODEL}): {last_exc}"
    ) from last_exc


def extract_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


def normalize_action(action: dict) -> dict:
    if not isinstance(action, dict):
        raise ValueError("AI did not return a usable JSON action object.")

    name = str(action.get("action", "")).strip().lower()
    aliases = {
        "open": "navigate", "visit": "navigate", "go": "navigate",
        "type": "fill", "input": "fill", "enter": "fill", "search": "fill",
        "goback": "back", "go_back": "back", "previous": "back",
    }
    name = aliases.get(name, name)

    if name not in SUPPORTED_ACTIONS:
        raise ValueError(f"Unsupported AI action: {name}")

    return {
        "action": name,
        "target": str(action.get("target", "") or "").strip(),
        "value": str(action.get("value", "") or ""),
        "reason": str(action.get("reason", "") or "").strip(),
    }


def valid_targets(observation) -> set:
    """Every exact selector string the model was allowed to choose from."""
    elements = (observation or {}).get("elements") or []
    return {_selector_for_element(el) for el in elements}


def target_is_grounded(action: dict, observation) -> bool:
    """True if this action's target is either not needed, or matches a real
    element from the current snapshot. navigate/wait/back/scroll (no target
    required) always pass."""
    if action["action"] not in {"click", "fill", "press", "select", "extract_text"}:
        return True
    if not action["target"]:
        return False
    return action["target"] in valid_targets(observation)


def _normalize_url(u: str) -> str:
    u = (u or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u.rstrip("/")


def is_noop_navigate(action: dict, observation) -> bool:
    """True if this is a 'navigate' to the exact URL we're already on — the
    agent got stuck re-issuing the same navigation instead of doing
    something new."""
    if action.get("action") != "navigate":
        return False
    current = _normalize_url((observation or {}).get("url", ""))
    target = _normalize_url(action.get("value") or action.get("target") or "")
    return bool(current) and bool(target) and current == target


def _recently_invalid_targets(history, limit=15) -> list:
    """Targets that have already been confirmed NOT to exist on some recent
    page (either a hallucinated selector or a no-op navigate). Collected so
    we can tell the model 'don't propose these again' up front, instead of
    relying on it to notice the pattern in raw history."""
    seen = []
    for entry in (history or [])[-limit:]:
        err = str(entry.get("error", ""))
        target = entry.get("target")
        if target and ("does not exist on the current page" in err or "already on this URL" in err):
            if target not in seen:
                seen.append(target)
    return seen


def _selector_for_element(el: dict) -> str:
    """Build the exact selector string the executor will accept for one snapshot element."""
    if el.get("id"):
        return f"#{el['id']}"
    if el.get("name"):
        return f"[name=\"{el['name']}\"]"
    if el.get("aria_label"):
        return f"[aria-label=\"{el['aria_label']}\"]"
    if el.get("placeholder"):
        return f"[placeholder=\"{el['placeholder']}\"]"
    if el.get("text"):
        return el["text"]
    return f"({el.get('tag', 'element')} #{el.get('index')}, no stable selector)"


def resolve_index_target(action: dict, observation) -> dict:
    """The model is instructed to give a plain element INDEX NUMBER (e.g.
    "5") rather than reproducing a selector string, because small models
    reliably pick a number but unreliably reproduce an exact string — they
    tend to reconstruct a plausible-looking selector from memory instead of
    copying the real one (e.g. inventing "a[aria-label='Store']" for a link
    that has no aria-label at all). This resolves that index back to the
    real selector before grounding/execution.

    If the model ignores the instruction and returns a real selector string
    instead, this is a no-op and grounding still checks it normally."""
    if action["action"] not in {"click", "fill", "press", "select", "extract_text"}:
        return action
    target = action["target"]
    if not target or not target.strip().lstrip("-").isdigit():
        return action
    idx = int(target.strip())
    for el in (observation or {}).get("elements") or []:
        if el.get("index") == idx:
            resolved = dict(action)
            resolved["target"] = _selector_for_element(el)
            return resolved
    return action


def _format_selector_menu(observation) -> str:
    """Render the snapshot's interactive elements as a numbered menu. The
    model picks the [index] number for click/fill/press/select targets —
    it must NOT write its own selector string."""
    elements = (observation or {}).get("elements") or []
    if not elements:
        return "(no interactive elements detected on this page)"
    lines = []
    for el in elements:
        label = el.get("text") or el.get("aria_label") or el.get("placeholder") or el.get("name") or ""
        dest = f" -> {el['href']}" if el.get("href") else ""
        lines.append(f'[{el.get("index")}] <{el.get("tag")}> "{label[:60]}"{dest}')
    return "\n".join(lines)


def decide_next_action(user_prompt, observation, history, visited_urls=None):
    visited_urls = visited_urls or []
    selector_menu = _format_selector_menu(observation)
    current_url = (observation or {}).get("url", "")
    current_title = (observation or {}).get("title", "")
    avoid = _recently_invalid_targets(history)
    avoid_block = (
        f"\nINDEXES/URLS ALREADY CONFIRMED INVALID — do NOT pick these again:\n{json.dumps(avoid, indent=2)}\n"
        if avoid else ""
    )
    prompt = f"""
You are an autonomous browser automation agent.
Return EXACTLY ONE next action for the current browser state.

USER GOAL:
{user_prompt}

CURRENT URL: {current_url}
CURRENT PAGE TITLE: {current_title}

URLS ALREADY VISITED THIS TASK (you have NOT been anywhere else):
{json.dumps(visited_urls, indent=2)}
{avoid_block}
CLICKABLE/FILLABLE ELEMENTS ON THIS PAGE, each with an [index] number.
For <a> links, the destination URL (if known) is shown after "->":
{selector_menu}

RECENT HISTORY:
{json.dumps(history[-8:], indent=2, ensure_ascii=False)}

Rules:
- Continue from the current state; do not restart unnecessarily.
- For click/fill/press/select, set "target" to ONLY the [index] number shown above, as a plain string like "5". Do NOT write a CSS selector, aria-label, id, or any attribute string yourself — you cannot see the real HTML, only this list, so anything you write yourself will be wrong. Just copy the number.
- Do NOT choose "navigate" to CURRENT URL — you are already there; that action does nothing. Pick a different action (click a link/button, fill a field, scroll, etc).
- Do NOT pick any index or URL listed under INDEXES/URLS ALREADY CONFIRMED INVALID above — those have already failed and will fail again.
- Compare USER GOAL to URLS ALREADY VISITED. If the goal names a website (e.g.
  a company or site name) whose domain does not appear in URLS ALREADY
  VISITED, that website has NOT been opened yet — do not treat searching for
  its name as equivalent to visiting it. Look for a link in the element list
  whose "->" destination or text matches that site and click its [index] (or
  use "navigate" straight to that domain if no matching link is visible).
- If nothing in the element list matches what you need, use "scroll" or "wait" — never invent an index or selector.
- To type text into an <input> or <textarea>, use action "fill" with the text in "value". Do NOT use "click" on a text field unless you only need to focus it with no text to enter — "click" never types anything.
- After "fill"-ing a search box, the next action should normally be "press" with value "Enter" on that same [index] to submit it — filling alone does not submit anything.
- If your last action already filled this exact [index] with this exact value, do NOT fill it again with different text unless the goal explicitly asks for a new search on the SAME site — first check whether you should instead be navigating onward per the rule above.
- Never bypass CAPTCHA, bot checks, OTP, 2FA, or security verification.
- If a human verification/security challenge is visible, return action "wait" with reason "Human intervention required".
- If the goal is complete, return "done".
- Return only one action, as JSON only, no other text.
- Supported actions: navigate, click, fill, press, wait, back, scroll, select, extract_text, done.

Respond with ONLY a JSON object shaped like this (these field names are fixed;
the VALUES are yours to fill in based on the real page above, never copied
from this shape):
{{"action": "<one of the supported actions>", "target": "<[index] number as a string, e.g. \\"5\\", or empty>", "value": "<text to type, or empty>", "reason": "<short reason>"}}
"""
    data = extract_json(ask_gemma(prompt))
    action = normalize_action(data)
    return resolve_index_target(action, observation)


def replan(user_prompt, observation, history=None, current_url="", failed_action=None, visited_urls=None):
    history = history or []
    failed_action = failed_action or {}
    visited_urls = visited_urls or []
    selector_menu = _format_selector_menu(observation)
    current_title = (observation or {}).get("title", "")
    avoid = _recently_invalid_targets(history)
    if failed_action.get("target") and failed_action["target"] not in avoid:
        avoid.append(failed_action["target"])
    avoid_block = (
        f"\nTARGETS/URLS ALREADY CONFIRMED INVALID — do NOT pick these again:\n{json.dumps(avoid, indent=2)}\n"
        if avoid else ""
    )
    prompt = f"""
You are recovering an autonomous browser agent after an action failed.
Continue from the CURRENT browser state. Do not restart unnecessarily.

USER GOAL:
{user_prompt}
CURRENT URL: {current_url}
CURRENT PAGE TITLE: {current_title}

URLS ALREADY VISITED THIS TASK (you have NOT been anywhere else):
{json.dumps(visited_urls, indent=2)}
{avoid_block}
CLICKABLE/FILLABLE ELEMENTS ON THIS PAGE, each with an [index] number.
For <a> links, the destination URL (if known) is shown after "->":
{selector_menu}

FAILED ACTION (this target/index does not exist on this page — do not repeat it):
{json.dumps(failed_action, indent=2)}
HISTORY:
{json.dumps(history[-8:], indent=2, ensure_ascii=False)}

Return exactly ONE replacement action.
- For click/fill/press/select, set "target" to ONLY the [index] number shown above, as a plain string like "5". Do NOT write a CSS selector, aria-label, id, or any attribute string yourself — anything you construct yourself will be wrong. Just copy the number.
- Do NOT choose "navigate" to CURRENT URL — you are already there.
- Do NOT pick any index or URL listed under TARGETS/URLS ALREADY CONFIRMED INVALID above.
- If the goal names a website whose domain is not in URLS ALREADY VISITED, look for a matching link in the element list and click its [index], or "navigate" straight to that domain.
- If nothing in the element list matches what you need, use "scroll" or "wait" — never invent an index or selector.
- To type text into an <input> or <textarea>, use action "fill" with the text in "value", not "click".
- Never bypass CAPTCHA, bot checks, OTP, 2FA, or security verification.
- Supported actions: navigate, click, fill, press, wait, back, scroll, select, extract_text, done.

Respond with ONLY a JSON object shaped like this (field names fixed, values yours):
{{"action": "<one of the supported actions>", "target": "<[index] number as a string, e.g. \\"5\\", or empty>", "value": "<text to type, or empty>", "reason": "<short reason>"}}
"""
    data = extract_json(ask_gemma(prompt))
    action = normalize_action(data)
    return resolve_index_target(action, observation)