
![](CHAL.png)

<h1 align="center">
  C-HAL: Council of Hierarchical Agentic Language
</h1>

**C-HAL** (pronounced "kal") is a modular framework for orchestrating structured dialectic debates between multiple large language model (LLM) agents. Each agent represents a distinct philosophical, logical, or epistemological position. The system manages multi-stage debates  that consist of challenges, rebuttals, adjudication, and synthesis, with the goal of resolving disagreements, generating insight, refining collective understanding, and work toward a unified internal belief convergence among the agents.

---

## Purpose

C-HAL serves as both a philosophical exploration tool as well as an active direction of research in agentic AI. It simulates how a council of intelligent agents might reason, argue, and converge toward truth when equipped with different assumptions and reasoning styles - partly inspired by complex reasoning systems in brains and complex societies, similar to Minsky's "Society of Mind".

It is designed for:
- Researchers studying multi-agent alignment, reasoning, and analyzing the evolution of complex belief systems in AI
- Philosophers interested in formalized dialectics
- Prompt engineers building complex agent architectures
- Developers exploring LLM debates and optimization in latent prompt spaces

---

## Core Features

- **Multi-Stage Debate Orchestration**  
  Includes briefing, opening positions, critique generation, rebuttals, adjudication, belief updates, and a final synthesis for clear expositional prose.

- **Agent Prompt Architecture**  
  Agents are instantiated via a combination of a *universal prompt* and a *persona-specific prompt*, allowing for structured diversity.

- **Belief Trajectories & Visualization**  
  Track agent belief embeddings over the course of the dialect using attention-driven (BERT-like) embeddings and dimensionality reduction techniques (e.g., UMAP).

- **Support for Multiple LLM Providers**  
  Easily integrate OpenAI, Anthropic, and other model APIs.

- **Extensible Pipeline**  
  Each stage is modular and customizable for research or philosophical use cases.

---

## Installation

> **Note:** Python 3.10+ and [Poetry](https://python-poetry.org/) 2.1.3+ are required.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/c-hal.git
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

## 🛠 Technologies Used

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



