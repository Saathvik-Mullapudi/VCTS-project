# Project Specific Rules

## Smoke Testing
Before proposing a commit, wrapping up a task, or declaring changes complete, you MUST:
1. Run the smoke test using the `run_command` tool (e.g. `python main.py 2>&1 | Select-String -Pattern "crossings|End of stream|Done!|Error|Traceback|Exception" | Select-Object -First 10`).
2. Verify that the application behavior matches the checklist in `docs/testing_checklist.md`.
3. Inform the user of the smoke test results to guarantee no regressions were introduced.

## Commit Message Style
When asked for a commit, give a heading and only 1 or 2 paragraphs explaining what it does.
