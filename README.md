
<p align="center">
  <img src="C-HAL.png" alt="CHAL Logo" width="300"/>
</p>

<h1 align="center">
  C-HAL: Council of Hierarchical Agentic Language
</h1>



**C-HAL** (pronounced "kal") is a framework for orchestrating structured dialectic debates between multiple large language model (LLM) agents. Each agent represents a distinct philosophical, logical, or epistemological position. The system manages multi-stage debates that consist of challenges, rebuttals, adjudication, and synthesis, with the goal of resolving disagreements, generating insight, refining collective understanding, and working toward a unified internal belief convergence among the agents.

This repository serves as the official open-source implementation of C-HAL. For more thorough details regarding the scientific development and workings of C-HAL, please see the paper (**INSERT PAPER HERE**).

---


## Table of Contents

- [Purpose](#purpose)
- [Core Features](#primary-features)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Poetry](#2-install-poetry)
  - [3. Install Project Dependencies](#3-install-project-dependencies)
  - [4. Activate the Environment](#4-activate-the-environment)
- [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Future Plans](#future-plans)
- [Technologies Used](#python-libraries-used)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Purpose

C-HAL serves as both a philosophical exploration tool as well as an active direction of research in agentic AI. It simulates how a council of intelligent agents might reason, argue, and converge toward truth when equipped with different assumptions and reasoning styles - partly inspired by complex reasoning systems in the human brain as well as complex societies, similar to Minsky's "Society of Mind".

It is designed for:
- Researchers studying multi-agent alignment, reasoning, and the evolution of complex belief systems in AI
- Philosophers interested in formalized dialectics as well as thoughtful exploration of deep, enigmatic, and unanswered questions at the heart of humanity
- Prompt engineers building complex agentic architectures
- Developers exploring LLM capabilities and optimization in latent prompt spaces

---

## Primary Features

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

> **Note:** Python 3.10+ and [Poetry](https://python-poetry.org/) 2.1.3+ are required.

### 1. Clone the Repository

```bash
git clone https://github.com/GdKent/C-HAL.git
cd c-hal
```

### 2. Install Poetry

Use the official installer:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Verify the installation:

```bash
poetry --version
# Should output: Poetry (version 2.1.3)
```

### 3. Install Project Dependencies

```bash
poetry install
```

This will install all required packages into a virtual environment automatically managed by Poetry.

### 4. Activate the Environment

```bash
poetry shell
```

You can now run the project with standard Python commands.

---

## Quick Start


---


## Running Tests

Tests will be placed in the `tests/` folder. To run them:

```bash
poetry run pytest
```

---

## Project Structure

```text
C-HAL/
├── .github/              # GitHub workflows, issue templates, etc.
├── docs/                 # Documentation and design notes
├── src/
│   └── chal/             # Core Python package code
├── tests/                # Unit and integration tests
├── README.md             # Project description and usage
├── pyproject.toml        # Poetry config and dependencies
├── poetry.lock           # Poetry lockfile (auto-generated)
├── LICENSE               # Project license (MIT)
└── .gitignore            # Files and folders to exclude from version control
```

---

## Documentation

Documentation is currently under development and will live in the `docs/` directory.

Planned docs:
- Agent prompting strategies
- Debate pipeline stages
- Adjudication subprotocols
- Visualization tooling
- Optimizing in latent prompt space (experimental)

---

## Future Plans

- Latent space prompt optimization
- Dialectic score metrics (truth-seeking, novelty, coherence)
- Richer visualizations of debate structure
- Interactive front-end for public debates
- Auto-generated expository essays from debates

---

## Python Libraries Used

- **Python 3.10+**
- **Poetry** for dependency management
- **OpenAI API** (GPT-4, GPT-4o, etc.)
- **tiktoken** for token accounting
- **UMAP**, **matplotlib**, and **scikit-learn** for visualization

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## Contributing

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



