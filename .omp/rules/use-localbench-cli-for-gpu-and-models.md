---
name: use-localbench-cli-for-gpu-and-models
description: "Check GPU, model state and test runs through this repo's localbench CLI, never raw ollama/curl calls or --allow-busy"
condition: ["--allow-busy", "\\bollama (ps|list|stop|pull)\\b", "curl[^\\n]{0,120}127\\.0\\.0\\.1:11434"]
scope: "tool:bash"
---

Use the repo's own CLI (`localbench`, installed from this checkout with `uv tool install -e .`) for every GPU and model action. It records what it does and refuses a busy machine; raw calls do neither.

- Resident models and GPU use: `localbench status` or `localbench gpu --seconds 5`, not `ollama ps` or curl to :11434.
- Keeping local models off the GPU: `localbench park`, then `localbench park --status`; afterwards `localbench unpark`.
- Installed models and cleanup: `localbench models` and `uv run python scripts/prune_models.py`, not `ollama list` or `ollama rm`.
- Every test run, smokes included: `localbench run|aa|ab ... --wait-idle 1800`, so preflight confirms the GPU is quiet before it starts. Never pass `--allow-busy`; a busy GPU means park or wait, not bypass.
- If the CLI lacks something you need, add it to the CLI with a test instead of working around it with raw ollama or curl.