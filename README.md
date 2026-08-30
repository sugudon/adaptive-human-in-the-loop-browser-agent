# Adaptive Human-in-the-Loop Browser Agent

> **Automate everything that can be automated. Interrupt humans only when they are actually needed.**

An agentic browser automation system combining **LLMs, Playwright, FastAPI, and human-in-the-loop intervention**.

The agent follows:

```text
OBSERVE → DECIDE → EXECUTE → OBSERVE → ...
```

It operates autonomously for routine browser tasks and pauses only when human identity, authorization, judgment, or intervention is genuinely required.

## Why this project?

Traditional automation follows a fixed workflow:

```text
Step 1 → Step 2 → Step 3 → Step 4
```

This project adapts to the current browser state:

```text
                    USER GOAL
                        │
                        ▼
                    OBSERVE
                        │
                        ▼
                     DECIDE
                      (LLM)
                        │
                        ▼
                   RISK CHECK
                   /         \
                  /           \
             SAFE             HUMAN NEEDED
              │                     │
              ▼                     ▼
           EXECUTE                PAUSE
              │                     │
              ▼                     ▼
           OBSERVE             HUMAN TAKEOVER
              │                     │
              └──────────┬──────────┘
                         ▼
                      OBSERVE
```

## Core idea

The user gives a high-level goal:

```text
Open Google.com, search for Amazon, open the Amazon website,
and search for laptop.
```

The user does not need to describe every click. The agent determines the next action from the current browser state.

## Human-in-the-Loop

Human interaction is an **exception mechanism**, not an approval step for every action.

### Autonomous

- Navigation
- Searching
- Clicking
- Form filling
- Scrolling
- Reading page content
- Waiting
- Pagination
- Applying filters
- Normal downloads
- Recovery and replanning

### Human intervention

- CAPTCHA
- "Are you human?" verification
- OTP / verification code
- Two-factor authentication
- Security challenges
- Payment authorization
- High-impact or irreversible actions
- Ambiguous decisions
- Explicit user takeover

The agent should never attempt to bypass CAPTCHA or other human-verification mechanisms.

## Real-world use cases

### 1. Enterprise IT Operations

```text
Check device warranty
→ Find driver
→ Download
→ Install
→ Verify
```

If a vendor portal requests human verification, the agent pauses and waits for the operator.

### 2. Recruitment Automation

```text
Search jobs
→ Filter location
→ Read descriptions
→ Identify suitable roles
→ Fill application
→ Upload resume
```

If a question requires personal judgment, the agent requests input and continues.

### 3. Enterprise Legacy Applications

Useful for internal systems with no API, old interfaces, poor documentation, or repetitive manual workflows.

```text
Open HR portal
→ Find employee
→ Update information
→ Submit
```

### 4. E-commerce Research

```text
Find laptops under ₹80,000
→ Compare specifications
→ Filter results
→ Prepare shortlist
```

A final purchase can require human authorization.

### 5. Travel Research and Booking

```text
Search flights
→ Compare prices
→ Filter baggage
→ Compare schedules
```

The agent can pause before final payment or booking.

### 6. SaaS Administration

```text
Read employee list
→ Create accounts
→ Assign standard roles
```

Privileged access changes can require human intervention.

## Risk-based autonomy

Not every action has the same risk.

| Action | Risk | Behavior |
|---|---|---|
| Open website | Low | Automatic |
| Search | Low | Automatic |
| Scroll | Low | Automatic |
| Read page | Low | Automatic |
| Fill search field | Low | Automatic |
| Download document | Medium | Policy-based |
| Login | Medium | Policy-based |
| Fill sensitive information | High | Policy-based |
| CAPTCHA | High | Human |
| OTP | High | Human |
| Delete account | High | Human |
| Place order | High | Human |
| Transfer money | High | Human |

## Architecture

```text
┌─────────────────────────────────────────────┐
│                  USER GOAL                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │    OBSERVE     │
              │ Browser / DOM  │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │     DECIDE     │
              │      LLM       │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │   RISK CHECK   │
              └───────┬────────┘
                      │
              ┌───────┴────────┐
              ▼                ▼
          SAFE ACTION      HUMAN NEEDED
              │                │
              ▼                ▼
          PLAYWRIGHT          PAUSE
              │                │
              ▼                ▼
          BROWSER         HUMAN TAKEOVER
              │                │
              └───────┬────────┘
                      ▼
                   OBSERVE
```

## Agent loop

```python
while task_not_complete:

    observation = observe_browser()

    decision = llm_decide(
        goal,
        observation,
        history
    )

    if human_intervention_required(observation, decision):
        pause_agent()
        wait_for_human()
        resume_agent()
        continue

    result = execute(decision)

    if execution_failed(result):
        reobserve()
        replan()

    if task_complete():
        finish()
```

The agent makes **one next decision at a time** instead of generating a long brittle workflow in advance.

## Example

### User

```text
Open Google.com and search for Amazon laptops.
```

### Agent

```text
OBSERVE
↓
URL: about:blank

DECIDE
↓
Navigate to https://google.com

EXECUTE
↓
Google loaded

OBSERVE
↓
Search input detected

DECIDE
↓
Fill search input with "Amazon laptops"

EXECUTE
↓
Search completed

OBSERVE
↓
Search results detected

DECIDE
↓
Continue...
```

## Human intervention example

If a website displays:

```text
Are you human?

[CAPTCHA]
```

the agent pauses:

```text
OBSERVE
    ↓
Human verification detected
    ↓
PAUSE
    ↓
Human completes verification
    ↓
RESUME
    ↓
OBSERVE
    ↓
DECIDE
    ↓
EXECUTE
```

The agent does not need to know exactly what the human did. It obtains a fresh browser observation and continues.

## Recovery and replanning

Dynamic websites can change.

An element may:

- disappear
- move
- change text
- become hidden
- load later
- be replaced

The agent should recover:

```text
Element not found
       ↓
Observe again
       ↓
LLM re-evaluates
       ↓
Find alternative
       ↓
Execute
```

## Technology Stack

- **Python**
- **FastAPI**
- **Streamlit**
- **Playwright**
- **Ollama**
- **Gemma**
- **Pydantic**
- Agentic AI
- Human-in-the-loop
- Risk-based autonomy
- Replanning

## Project Structure

```text
Adaptive-Human-in-the-Loop Browser-Agent/
│
├── app.py
├── main.py
│
├── agent/
│   ├── __init__.py
│   ├── agent.py
│   └── state.py
│
└── automation/
    ├── __init__.py
    └── executor.py
```

## Installation

### 1. Create virtual environment

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn streamlit playwright requests pydantic
```

### 3. Install Playwright browser

```bash
playwright install chromium
```

### 4. Install and configure Ollama

Pull the model:

```bash
ollama pull gemma2:2b
```

Verify:

```bash
ollama list
```

The default Ollama endpoint is:

```text
http://localhost:11434
```

## Running

Start FastAPI:

```bash
uvicorn main:app --reload --port 8000
```

In another terminal:

```bash
streamlit run app.py
```

## Example tasks

Start simple and increase complexity.

### Level 1

```text
Open Google.com.
```

### Level 2

```text
Open Google.com and search for Amazon.
```

### Level 3

```text
Open Google.com, search for Amazon, and open the Amazon website.
```

### Level 4

```text
Open Google.com, search for Amazon, open the Amazon website, and search for laptop.
```

### Level 5

```text
Find laptops under ₹80,000 and prepare a shortlist.
```

## Development Roadmap

### Phase 1 — Browser Agent

- [x] Browser automation
- [x] Browser observation
- [x] LLM-based next-action decision
- [x] Playwright execution
- [x] Task state
- [x] Action history

### Phase 2 — Agent Reliability

- [ ] Better DOM extraction
- [ ] Robust element identification
- [ ] Automatic retries
- [ ] Re-observation after failures
- [ ] LLM replanning
- [ ] Page-change detection
- [ ] Loop detection

### Phase 3 — Human-in-the-Loop

- [ ] CAPTCHA detection
- [ ] OTP detection
- [ ] Security challenge detection
- [ ] Human takeover
- [ ] Pause/resume
- [ ] User interrupt
- [ ] Skip action
- [ ] Replay action
- [ ] Edit action

### Phase 4 — Risk Engine

- [ ] Action risk classification
- [ ] Policy engine
- [ ] High-risk action detection
- [ ] Human authorization
- [ ] Organization-specific policies
- [ ] Audit logging

### Phase 5 — Production Agent

- [ ] Persistent sessions
- [ ] Multi-tab support
- [ ] File upload/download handling
- [ ] Authentication state management
- [ ] Agent memory
- [ ] Task scheduling
- [ ] Observability
- [ ] Metrics
- [ ] Secure credential handling
- [ ] Multi-agent workflows

## Security Principles

The agent follows a **human-controlled autonomy model**.

It should not:

- Bypass CAPTCHA
- Circumvent security controls
- Guess OTPs
- Attempt to defeat authentication
- Perform unauthorized transactions
- Make irreversible high-impact decisions without appropriate authorization

Instead:

```text
Security boundary detected
        ↓
Pause
        ↓
Human intervention
        ↓
Resume
```

## Why this is Agentic AI

Traditional automation:

```text
Fixed workflow
    ↓
Execute
```

LLM-assisted automation:

```text
Prompt
    ↓
Generate steps
    ↓
Execute
```

This project uses an agentic loop:

```text
Goal
 ↓
Observe environment
 ↓
Reason about current state
 ↓
Choose next action
 ↓
Execute
 ↓
Observe changed environment
 ↓
Re-evaluate
 ↓
Recover / Replan
 ↓
Continue
```

The browser environment is therefore part of the decision loop.

## Portfolio Value

This project demonstrates practical experience with:

- Generative AI
- LLM integration
- Agentic AI
- Browser automation
- Playwright
- Python
- FastAPI
- Streamlit
- Ollama
- Local LLMs
- Prompt engineering
- Tool execution
- Replanning
- Human-in-the-loop systems
- Risk-based autonomy
- State management

## Future Vision

The long-term goal is not:

> "AI that clicks buttons."

The goal is:

> **An autonomous digital worker that can operate software through the browser, adapt to changing environments, recover from routine failures, and intelligently involve humans only when human judgment, identity, authorization, or intervention is required.**

```text
                 DIGITAL WORKER
                       │
              ┌────────┴────────┐
              │                 │
          AUTONOMOUS          HUMAN
           WORKFLOW          INTERVENTION
              │                 │
              └────────┬────────┘
                       │
                 TASK COMPLETED
```

## License

No license has been granted for this project. All rights are reserved by the copyright holder. You may view the source code for reference, but permission is required to copy, modify, distribute, or reuse the code.
