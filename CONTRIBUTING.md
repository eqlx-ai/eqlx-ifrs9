# Contributing to eqlx-ifrs9

Thanks for helping. Bug reports, new calculations, rule profiles for other jurisdictions and
documentation fixes are welcome.

## Ground rules

1. **Never share real loan data.** No real borrowers, account numbers or balances in issues, tests or
   examples. Describe the case, or reproduce it with `ifrs9 sample` or a few made-up rows.
2. **Deterministic only.** A calculation is a written formula with explicit parameters: no model
   fitted inside, no randomness, no hidden default. Same input, same output, independent of row order.
3. **Independent truth.** Every calculation is tested against something that does not share its
   code: a published example, a closed form, direct enumeration or simulation. A test that repeats
   the implementation's own expression does not count.
4. **Public sources.** Cite the paragraph of the standard, the regulation article or the paper a
   rule follows. Nothing from confidential or proprietary material.
5. **Choices are settings.** Where the standard leaves a choice, add a parameter and document it;
   do not pick for the user.
6. **Dependencies:** NumPy and SciPy for the core, `mcp` for the server. Propose anything else in an
   issue first.

## Workflow

1. Open an issue first for anything larger than a small fix.
2. Fork, create a branch (`fix/...`, `feat/...`), and run the checks locally:

   ```bash
   uv sync --all-extras
   uv run pytest
   uv run ruff check . && uv run ruff format --check . && uv run mypy src
   ```

3. Open a pull request against `main` and fill in the template. CI must pass before review; for
   outside contributors a maintainer approves the CI run.

By contributing you agree that your contribution is licensed under the Apache License 2.0.

Questions: open an issue or write to research@eqlx.ai.
