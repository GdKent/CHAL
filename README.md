
<p align="center">
  <img src="CHAL.png" alt="CHAL Logo" width="300"/>
</p>

<h1 align="center">
  CHAL: Council of Hierarchical Agentic Language
</h1>



**CHAL** (pronounced "kal") is a framework for orchestrating structured dialectic debates between multiple large language model (LLM) agents. Each agent represents a distinct philosophical, logical, or epistemological position. The system manages multi-stage debates that consist of challenges, rebuttals, adjudication, and synthesis, with the goal of resolving disagreements, generating insight, refining collective understanding, and working toward a unified internal belief convergence among the agents.

This repository serves as the official open-source implementation of CHAL. For more thorough details regarding the scientific development and workings of CHAL, please see the paper "CHAL: Council of Hierarchical Agentic Language".

---


## Table of Contents

- [Purpose](#purpose)
  - [Primary Features](#primary-features)
- [Installation](#installation)
  - [0. Prerequisites](#0-prerequisites)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Poetry](#2-install-poetry)
  - [3. Install Project Dependencies](#3-install-project-dependencies)
  - [4. Verify Installation](#5-verify-installation)
  - [5. Run the Test Suite (Optional)](#6-run-the-test-suite-optional)
  - [6. Updating Your Environment (Optional)](#7-updating-your-environment-optional)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Purpose

CHAL serves as both a philosophical exploration tool as well as an active direction of research in agentic AI. It simulates how a council of intelligent agents might reason, argue, and converge toward truth when equipped with different assumptions and reasoning styles - partly inspired by complex reasoning systems in the human brain as well as complex societies, similar to Minsky's "Society of Mind".

It is designed for:
- Researchers studying multi-agent alignment, reasoning, and the evolution of complex belief systems in AI
- Philosophers interested in formalized dialectics as well as thoughtful exploration of deep, enigmatic, and unanswered questions at the heart of humanity
- Prompt engineers building complex agentic architectures
- Developers exploring LLM capabilities and optimization in latent prompt spaces

### Primary Features

- **Structured Multi-Stage Debate Orchestration**  
  Includes univeral briefing, opening positions, critique generation, rebuttals, adjudication, internal belief updates, and a final synthesis for clear expositional prose.

- **Agent Prompt Architecture**  
  Agents are instantiated via a combination of a *universal prompt* and a *persona-specific prompt*, allowing for structured diversity. Each agent maintains a current internal belief system that is altered throughout the course of the dialect.

- **Belief Trajectories & Visualization**  
  Track agent internal belief embeddings over the course of the dialect using attention-based (BERT-like) embeddings along with dimensionality reduction techniques (e.g., UMAP).

- **Support for Multiple LLM Providers**  
  Easily integrate OpenAI, Anthropic, Google, and other model APIs.

---

## Installation

This section explains exactly how to download, install, and set up **CHAL** on your local machine. Follow these steps carefully to ensure that your environment is configured correctly.

> **Note:** Python 3.10+ and [Poetry](https://python-poetry.org/) 2.1.3+ are required.

#### 🖥️ Terminal and Command Conventions

Throughout this guide, here's how to interpret the command labels that are used:

| Label | Meaning | Where to Type It |
|--------|----------|------------------|
| `bash` | (Born Again SHell) A **system terminal** command | macOS/Linux Terminal, Windows PowerShell, or Git Bash |
| `python` | Python code run **inside** the Python interpreter | After typing `python` or `python3` |
| `poetry` | Poetry dependency or environment command | Terminal (inside your project folder) |

> **If you’re using Anaconda**, you’ll run commands in the **Anaconda Prompt** instead of a normal terminal.  
> It behaves the same way for almost all commands.


### 0. Prerequisites

Before you begin, make sure that you have the following installed:

- **Python 3.10 or later**  
  You can verify by opening your system terminal (macOS/Linux), Command Prompt (Windows), or Anaconda Prompt (Anaconda) within the desired directory (or cd to the desired directory) and running:
  ```bash
  python --version
  ```
  It should output something like:
  ```
  Python 3.10.13
  ```

- **Git**  
  You can check by running:
  ```bash
  git --version
  ```
  You should see a version number such as:
  ```
  git version 2.44.0
  ```

If you do not have Python or Git installed, please install them first:

- **Windows:** [Download Python](https://www.python.org/downloads/windows/) and [Git for Windows](https://git-scm.com/download/win)
- **macOS:** Use Homebrew (`brew install python git`)
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip git
  ```

---

### 1. Clone the Repository

Open your **terminal** (macOS/Linux), **PowerShell or Git Bash** (Windows), or **Anaconda Prompt** (if using Anaconda).  
Once you navigate to the desired directory that you would like to work out of, then type:

```bash
git clone https://github.com/GdKent/CHAL.git
cd CHAL
```

This will download the repository into your desired working directory and move you into the CHAL directory.


### 2. Install Poetry

CHAL is distributed as a modular Python package using [Poetry](https://python-poetry.org/) for clean dependency and environment management. Poetry will ensure that all collaborators will have the same package versions and isolated environments.

#### 🟩 If You’re Using Standard Python:

Run this command in your **system terminal** (macOS/Linux) or **PowerShell or Git Bash** (Windows):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then close and reopen your terminal and verify that Poetry installed correctly:

```bash
poetry --version
```

You should see something like:
```
Poetry (version 2.1.3)
```

> ⚠️ **Windows Note:**  
> If `poetry` is not recognized, add Poetry’s installation path to your system PATH environment variable.  
> The installer usually tells you the correct path when it finishes.

---

#### 🟦 If You’re Using Anaconda:

Poetry works perfectly inside Anaconda, but create a dedicated environment first.

1. Open your **Anaconda Prompt**.
2. Create a new environment for CHAL:
   ```bash
   conda create -n chal_env python=3.10
   ```
3. Activate the environment:
   ```bash
   conda activate chal_env
   ```
4. Install Poetry inside this environment:
   ```bash
   pip install poetry==2.1.3
   ```
5. Verify installation:
   ```bash
   poetry --version
   ```
   Expected output:
   ```
   Poetry (version 2.1.3)
   ```

Now all future steps should be done inside this active conda environment.

---

### 3. Install Project Dependencies

Make sure you are **inside the project folder (`CHAL`)** and your Poetry (or conda + Poetry) environment is active.  
Then run:

```bash
poetry install
```

This command:
- Creates a virtual environment (.venv) automatically (if you’re not using conda)
- Installs all dependencies listed in `pyproject.toml`
- Locks dependency versions for reproducibility (`poetry.lock`)

Once finished, you’ll see something like:
```
Installing dependencies from lock file
Package operations: 12 installs, 0 updates, 0 removals
```

---


### 4: Verify Installation

To start using the environment Poetry created, use "poetry run" commands. For example, try importing the chal library in Python within the active environment:

```bash
poetry run python -c "import chal"
```

If no errors appear, your installation is successful.

> 💡 **Anaconda users:**  
> If you used `conda create` earlier, your environment is already active (you’ll see `(chal_env)` before your prompt), simply use `poetry run` commands.

---


### 5: Run the Test Suite (Optional) **(Under Development)**

To make sure everything is working as expected, run:

```bash
poetry run pytest
```

This will execute all tests inside the `tests/` directory.  
If you see something similar to:
```
================= 5 passed in 2.13s =================
```
everything is functioning correctly.

---


### 6: Updating Your Environment (Optional)

If the repository is updated with new code or dependencies, you can refresh your local setup by running the following in your **terminal** (macOS/Linux), **PowerShell or Git Bash** (Windows), or **Anaconda Prompt** (if using Anaconda):

```bash
git pull
poetry install
```

This pulls the latest changes and installs any new dependencies.

---

> 💬 If you encounter any installation issues, please open a GitHub issue here:  
> [https://github.com/GdKent/CHAL/issues](https://github.com/GdKent/C-HAL/issues)


## Project Structure **(Under Development)**

```text
CHAL/
├── .github/              # GitHub workflows, issue templates, etc. **(to be added)**
├── docs/                 # Documentation and design notes **(to be added)**
├── src/
│   └── chal/             # Core Python package code
├── tests/                # Unit and integration tests **(to be added)**
├── README.md             # Project description and usage
├── pyproject.toml        # Poetry config and dependencies
├── poetry.lock           # Poetry lockfile (auto-generated)
├── LICENSE               # Project license (MIT)
└── .gitignore            # Files and folders to exclude from version control
```

---

## Documentation <span style="color:red">(Under Development)</span>

Documentation is currently under development and will live in the `docs/` directory.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## Contributing **(Under Development)**

We welcome contributions, suggestions, and critiques.

To get started:
1. Fork this repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

Please see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more detailed guidelines.

---

## Contact

Created by **Griffin Dean Kent**. For questions, reach out via GitHub or [g.hal.dkent@gmail.com].

---



