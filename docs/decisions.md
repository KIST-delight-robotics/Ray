# Decision Log

## 2026-02-24 — Phase 0: Project Setup

- **Git strategy**: Orphan branch `revamped` on existing repo (`KIST-delight-robotics/Ray`). Clean history, no legacy baggage.
- **Package manager**: `uv` with `pyproject.toml`. No `requirements.txt`.
- **Dev tools**: `ruff` for linting + formatting, `pytest` for testing. All configured in `pyproject.toml`.
- **Python version**: `>=3.11` (required for modern typing features).
- **Directory structure**: Full skeleton created upfront per CLAUDE.md spec. All directories have `__init__.py` for proper package resolution.
