# Contributing to C-HAL

Thank you for your interest in contributing to **C-HAL**!

C-HAL is a modular framework for orchestrating structured debates between large language model (LLM) agents. Whether you're fixing bugs, proposing features, improving documentation, or crafting new prompt personas, we welcome your contributions.

---

## Getting Started

To contribute, follow these steps:

1. **Fork** the repository on GitHub and **clone** it locally:

   ```bash
   git clone https://github.com/YOUR_USERNAME/c-hal.git
   cd c-hal
   ```

2. **Create a new branch** for your feature or fix:

   ```bash
   git checkout -b feature/my-feature-name
   ```

3. **Install Poetry** (if you haven’t already):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

   Confirm installation:

   ```bash
   poetry --version
   ```

4. **Install dependencies**:

   ```bash
   poetry install
   ```

5. **Activate the virtual environment**:

   ```bash
   poetry shell
   ```

---

## Code Style Guidelines

We enforce consistent, clean code using these tools:

| Tool      | Purpose                      |
|-----------|------------------------------|
| `black`   | Code formatter                |
| `isort`   | Sorts and groups imports      |
| `mypy`    | Static type checking          |
| `pytest`  | Test framework                |

Before committing, **run these tools** from the root directory:

```bash
poetry run black .
poetry run isort .
poetry run mypy src/
```

---

## Testing

All code contributions should be accompanied by relevant tests when appropriate.

To run the full test suite:

```bash
poetry run pytest
```

If you’re adding a new module or function, add corresponding tests in the `tests/` directory.

---

## Project Structure Overview

```text
c-hal/
├── src/
│   └── chal/               # Core package logic
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation and guides
├── pyproject.toml          # Poetry config and dependencies
├── README.md               # Project overview
├── .gitignore              # Files to exclude from Git
└── LICENSE                 # MIT license
```

---

## Commit and PR Guidelines

Please follow these conventions to make collaboration smoother:

- Use clear and concise commit messages (e.g., `fix: correct UMAP label alignment` or `feat: add rebuttal parsing module`)
- Break large changes into smaller pull requests
- Document any public functions or modules
- Link to any relevant issues in your pull request description
- Tag reviewers if your PR needs attention
- Do not include unrelated formatting, commented-out code, or refactors in feature PRs

---

## Code of Conduct

We value inclusivity, respect, and constructive discussion. All contributors are expected to adhere to the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## Suggestions and Issues

Have a new idea for a prompt architecture, debate protocol, or visualization?  
Found a bug? Please open an issue or discussion on GitHub. Good first issues will be labeled as such.

---

Thank you again for helping to improve C-HAL.  
Your insights and creativity make the Council stronger.
