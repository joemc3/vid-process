# AGENTS.md

## Build / Lint / Test Commands
- **Run the monitor**: `python video_monitor.py`
- **Lint code**: `flake8 .` *(or `pylint $(git ls-files "*.py")`)*
- **Format code**: `black .` *(enforces 88‑column width, double quotes, trailing commas)*
- **Run all tests**: `pytest` *(install with `pip install pytest`)*
- **Run a single test**: `python -m unittest path/to/test_file.py::TestClass.test_method`

## Code‑Style Guidelines
- **Imports**: 3 groups, sorted alphabetically, separated by a blank line:
  ```python
  import os
  import sys

  from pathlib import Path
  import requests

  from mymodule import Foo
  ```
- **Formatting**: Use Black (default settings) and isort for import order.
- **Typing**: Add type hints for public functions and class methods. Prefer `Path` over `str` for filesystem paths.
- **Naming**:
  - Variables & functions: `snake_case`
  - Classes & Exceptions: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Error handling**: Catch specific exceptions, log with the standard `logging` module, and re‑raise if the caller must handle it.
- **Logging**: Use module‑level logger `log = logging.getLogger(__name__)` and include contextual info.
- **Docstrings**: Google style, one‑line summary followed by description and Args/Returns sections.
- **No magic numbers**: Define configuration values in `config.json` or module‑level constants.

## Additional Rules
- No `print` statements in production code – use `log.info/debug/warning/error` instead.
- Keep line length ≤ 88 characters (Black default).
- Prefer `Path` operations (`Path.is_file()`, `Path.stat().st_size`).
- Use f‑strings for string interpolation.
- Do not commit generated files (e.g., `__pycache__`, `*.pyc`).

*(No .cursor or Copilot rule files are present in this repository.)*