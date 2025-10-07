# Contributing to CHAL

Thank you for your interest in contributing to **CHAL**!

This document provides a complete guide on how to contribute to the project — including setup, branching, committing, pushing, and submitting pull requests.  
Please read through carefully before contributing your first code.

---

## 1. Getting Started

### 1.1 Fork or Clone the Repository

First, navigate to the desired directory that you wish to work in.

If you are an **external contributor**, fork the repo first using the **“Fork”** button on GitHub, then clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/CHAL.git
cd CHAL
```

If you are a **collaborator with push access**, clone the original repository directly:

```bash
git clone https://github.com/GdKent/CHAL.git
cd CHAL
```

> 💡 Tip: Run `git remote -v` afterward to confirm that your remote is correct.  
> You should see either your username’s fork or `GdKent/CHAL`.

---

## 🏗️ 2. Setting Up Your Development Environment

### 2.1 Verify Prerequisites

Make sure you have:
- **Python 3.10+**
- **Git** installed and configured
- **Poetry 2.1.3+**

You can verify by running:
```bash
python --version
git --version
poetry --version
```

If Poetry is missing, install it:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

> 🧠 Windows users: You can also use PowerShell or Anaconda Prompt.

---

### 2.2 Create or Activate a Virtual Environment

**If you are using Poetry (recommended):**

```bash
poetry install
```

**If you are using Anaconda:**

```bash
conda create -n chal python=3.10
conda activate chal
pip install poetry==2.1.3
poetry install
```

This installs all project dependencies defined in `pyproject.toml` and places you in an isolated environment.

---

## 🌿 3. Branching and Workflow

### 3.1 Main Branches

- **`main`** → The stable production branch. All official releases come from here.  
  No one pushes directly to `main` — only pull requests after review.

- **`dev`** → The development branch. All new features merge here first for testing and integration.

---

### 3.2 Create a Feature Branch

Before starting any new work, create a feature branch off of `dev`:

```bash
git checkout dev
git pull
git checkout -b feature/your-feature-name
```

Examples:
- `feature/add-debate-controller`
- `fix/prompt-encoding-bug`
- `docs/update-installation-guide`

Use clear, descriptive names for branches.

---

## 💻 4. Making Changes

### 4.1 Code Style

Please follow these standards before committing code:
- Use **Black** for formatting  
  ```bash
  poetry run black .
  ```
- Use **isort** to organize imports  
  ```bash
  poetry run isort .
  ```
- Use **mypy** for type checking  
  ```bash
  poetry run mypy src/
  ```

All source code should reside in:
```
src/chal/
```

If you add new modules, be sure to include docstrings explaining:
- Purpose
- Inputs and outputs
- Example usage if relevant

---

### 4.2 Running Tests

All tests live in the `tests/` directory. Before committing, run:

```bash
poetry run pytest
```

If you add new functionality, include corresponding tests.

> ✅ Passing tests are required before merging to `dev` or `main`.

---

## 🧾 5. Committing Your Changes

### 5.1 Add and Commit

Once your changes are ready:

```bash
git add .
git commit -m "feat: add belief update logic for stage 5"
```

Use **conventional commit messages**:

| Type | Purpose |
|------|----------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation update |
| `refactor:` | Code restructuring without behavior change |
| `test:` | Adding or updating tests |
| `chore:` | Minor maintenance or non-functional changes |

Examples:
- `feat: add adjudication module for Stage 4`
- `fix: correct token counting in prompt generator`
- `docs: update contributing guide for clarity`

---

### 5.2 Push Your Branch

Push your branch to GitHub:

```bash
git push origin feature/your-feature-name
```

If it’s your first push for that branch, Git will tell you the exact command to use (it’s the same as above).

---

## 🔀 6. Submitting a Pull Request (PR)

1. Go to your fork or the main CHAL repository on GitHub.
2. Click **Compare & pull request**.
3. Choose:
   - **Base branch:** `dev`
   - **Compare branch:** your feature branch
4. Add a clear title and description of what you changed.
5. Link to any relevant issue(s).
6. Submit the PR for review.

---

## 👀 7. Code Review Process

- Every PR must be reviewed by at least **one other collaborator** before merging.
- Reviewers will:
  - Check for correctness, clarity, and consistency.
  - Ensure that all tests pass.
  - Suggest improvements if necessary.
- Once approved, your PR will be merged into `dev`.
- Periodically, stable updates from `dev` will be merged into `main`.

---

## 🧹 8. Keeping Your Branch Up-to-Date

While working on a feature, keep your branch synchronized with `dev` to avoid merge conflicts:

```bash
git fetch origin
git checkout dev
git pull
git checkout feature/your-feature-name
git merge dev
```

Resolve any conflicts if prompted, then continue your work.

---

## 🧩 9. Project Structure Reference

```
CHAL/
├── src/
│   └── chal/              # Core package code
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation (optional)
├── README.md              # Main project overview
├── INSTALLATION.md        # Installation instructions
├── CONTRIBUTING.md        # This file
├── LICENSE                # MIT License
└── pyproject.toml         # Poetry configuration
```

---

## 🧠 10. Helpful Git Commands Summary

| Action | Command |
|--------|----------|
| Clone the repo | `git clone https://github.com/GdKent/CHAL.git` |
| Check current branch | `git branch` |
| Create a new branch | `git checkout -b feature/branch-name` |
| Switch branches | `git checkout branch-name` |
| Add files for commit | `git add .` |
| Commit changes | `git commit -m "commit message"` |
| Push branch | `git push origin feature/branch-name` |
| Fetch latest updates | `git fetch origin` |
| Merge dev into your branch | `git merge dev` |

---

## 🧭 11. After a Successful Merge

After your pull request is merged:
1. Switch back to `dev`:
   ```bash
   git checkout dev
   ```
2. Pull the latest changes:
   ```bash
   git pull
   ```
3. Delete your old branch (optional cleanup):
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

---

## 💬 12. Questions or Issues

If you encounter problems or have suggestions:
- Open an **Issue** in the GitHub repository.
- Use clear titles and descriptions (include error messages, screenshots, or logs if relevant).
- For general questions, use GitHub Discussions if enabled.

---

> 🧩 **In summary:**  
> - Work on a branch (never directly on `main`).  
> - Follow the coding standards.  
> - Run all tests before committing.  
> - Use clear, conventional commit messages.  
> - Submit PRs into `dev` for review and integration.  

Your contributions help make CHAL a rigorous, transparent, and collaborative platform for epistemic AI research.  
Thank you for helping build it!
