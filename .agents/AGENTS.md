# Project Specific Rules

## Smoke Testing
Before proposing a commit, wrapping up a task, or declaring changes complete, you MUST:
1. Run the smoke test using the `run_command` tool (e.g. `python main.py 2>&1 | Select-String -Pattern "crossings|End of stream|Done!|Error|Traceback|Exception" | Select-Object -First 10`).
2. Verify that the application behavior matches the checklist in `docs/testing_checklist.md`.
3. Inform the user of the smoke test results to guarantee no regressions were introduced.

## Commit Message Style
When asked for a commit, give a heading and only 1 or 2 paragraphs explaining what it does.

## Architecture & Code Quality (Anti-Spaghetti Rules)
When writing or modifying code in this project, you MUST strictly follow these principles:
1. **The Orchestrator Pattern:** `main.py` is an orchestrator. It must NEVER contain raw array slicing, mathematical geometry, or GStreamer buffer unpacking. It delegates to classes in `src/`.
2. **Separation of Concerns (One Job Rule):** Every class and file must do exactly one thing. Do not combine drawing logic with tracking, or tracking with inference.
3. **No Hidden State:** Do not use closures for callbacks. Do not pass giant dictionaries of state. State should be cleanly encapsulated in classes.
4. **Explicit Imports:** NEVER use wildcard imports (`from config import *`). Import exactly what is used so that names are traceable.
