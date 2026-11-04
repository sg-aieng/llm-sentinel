# Contributing to llm-sentinel

Thanks for helping improve llm-sentinel. Contributions should preserve the
project's fail-safe behavior: unavailable or unparseable security checks must
not silently become an `allow` verdict.

## Before you start

- Check existing issues and pull requests for related work.
- For security vulnerabilities, follow [SECURITY.md](SECURITY.md) instead of
  opening a public issue.
- Keep changes focused and explain behavior changes in the pull request.

## Development setup

The project requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Install the optional ML dependencies only when working on model-backed
inference:

```bash
python -m pip install -e ".[ml]"
```

The test suite disables model loading through `tests/conftest.py`, so ordinary
unit and integration tests do not require model downloads.

## Checks before submitting

Run the automated checks from the repository root:

```bash
python -m pytest -q
python -m compileall -q app sentinel tests examples
for script in e2e/*.sh; do bash -n "$script" || exit 1; done
git diff --check
```

The end-to-end scripts require a running server and exercise the REST API:

```bash
uvicorn app.main:app --port 8000
./e2e/run_all.sh
```

The e2e scripts assume the local server is running in a trusted environment.
When `SENTINEL_API_KEY` is configured, use an authenticated client or add the
matching bearer header to the scripts before running them.

## Code guidelines

- Keep deterministic checks cheap and place them before model-backed stages.
- Keep input and output behavior consistent between the SDK and REST API.
- Do not log prompt, output, secret, or PII content. Audit records should use
  hashes or explicitly approved identifiers only.
- Add regression tests for security-sensitive behavior, including both the
  blocking path and fail-safe behavior when a dependency is unavailable.
- Update the README and configuration documentation when public behavior or
  environment variables change.

## Pull requests

A pull request should include:

- a clear description of the problem and the chosen approach;
- tests covering new behavior and regressions;
- documentation for public API or configuration changes;
- notes about model, policy, performance, or compatibility implications.

Keep commits reviewable and avoid committing generated caches, virtual
environments, model weights, credentials, or `*.egg-info` build artifacts.
