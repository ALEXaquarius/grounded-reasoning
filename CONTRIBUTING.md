# Contributing

Thanks for your interest! This project follows a strict research methodology:
**proof before code, formal definitions, falsifiability, and honest reporting of
negative results.** Every capability in this library is backed by a theorem with
a numerical verification, not a heuristic.

## Development setup

```bash
git clone https://github.com/ALEXaquarius/grounded-reasoning
cd grounded-reasoning
pip install -e ".[dev]"
pytest tests/          # no API key needed — every LLM-dependent invariant is offline-locked
```

## Ground rules

1. **Every new capability needs a theorem + numerical verification** in
   `src/theory/theorems.py`, added to `ALL_THEOREMS`, with a corresponding test in
   `tests/test_theorems.py`.
2. **No secrets in the repo.** API keys are read from environment variables only
   (`.env` is git-ignored). Never hardcode a key or commit one. See
   [SECURITY.md](SECURITY.md).
3. **Tests must pass offline.** Anything that calls a real LLM must also have a
   deterministic offline lock (mock/simulated) so CI is green without keys.
4. **Be honest.** Negative results and limitations are welcome and expected —
   document them. Do not overclaim novelty; classical building blocks (Katz
   index, Neumann series, conformal prediction, Horn logic) are cited as such.

## Pull requests

- Keep changes focused; match the surrounding code style.
- Run `pytest tests/ -q` locally before opening a PR.
- Describe what you changed, why, and how it was verified.

## Public API

User-facing imports go through the `grounded_reasoning` package; internal
implementation lives under `src/`. Breaking changes to `grounded_reasoning/__init__.py`
require a version bump and a `CHANGELOG.md` entry.
