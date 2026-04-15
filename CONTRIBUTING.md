# Contributing

Thanks for helping improve `reliquary-inference`.

## Development Setup

```bash
cp env.example .env
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
```

## Contribution Rules

- keep protocol changes narrowly scoped and justified
- prefer tests for verifier, storage, chain-adapter, and task-source changes
- avoid committing host-specific details, wallet metadata, or runtime secrets
- update public docs when operator or public surfaces change

## Pull Requests

- explain the user-visible or operator-visible impact
- call out any protocol compatibility changes explicitly
- include validation notes

Use the pull request template in `.github/pull_request_template.md`.
