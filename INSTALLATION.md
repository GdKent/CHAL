
## Installation

This section explains exactly how to download, install, and set up **CHAL** on your local machine. Follow these steps carefully to ensure that your environment is configured correctly.

> **Note:** Python 3.10+ and [Poetry](https://python-poetry.org/) 2.1.3+ are required.

### 0. Prerequisites

Before you begin, make sure that you have the following installed:

- **Python 3.10 or later**  
  You can verify by opening your system terminal (macOS/Linux) or Command Prompt (Windows) within the desired directory (or cd to the desired directory) and running:
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
Then type:

```bash
git clone https://github.com/GdKent/CHAL.git
cd CHAL
```

This will download the repository and move you into its directory.


### 2. Install Poetry

CHAL is distributed as a modular Python package using [Poetry](https://python-poetry.org/) for clean dependency and environment management. Poetry will ensure that all collaborators will have the same package versions and isolated environments.

#### 🟩 If You’re Using Standard Python:

Run this command in your **system terminal** (macOS/Linux) or **PowerShell** (Windows):

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
- Creates a virtual environment automatically (if you’re not using conda)
- Installs all dependencies listed in `pyproject.toml`
- Locks dependency versions for reproducibility (`poetry.lock`)

Once finished, you’ll see something like:
```
Installing dependencies from lock file
Package operations: 12 installs, 0 updates, 0 removals
```

---

### 4. Activate the Virtual Environment

To start using the environment Poetry created:

```bash
poetry shell
```

You’ll see your terminal prompt change to something like:

```
(chal-py3.10) $
```

This means you’re now inside the CHAL environment.

> 💡 **Anaconda users:**  
> If you used `conda create` earlier, your environment is already active (you’ll see `(chal)` before your prompt).  
> You do **not** need to run `poetry shell` — stay in the conda environment and use `poetry run` commands instead.

---


### Step 5: Verify Installation

Run Python within the active environment:

```bash
python
```

Then type:
```python
import chal
chal.__version__
```

If no errors appear and a version number prints, your installation is successful.  
Exit Python:
```python
exit()
```

---


### Step 6: Run the Test Suite (Optional)

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


### Step 7: Updating Your Environment (Optional)

If the repository is updated with new code or dependencies, you can refresh your local setup:

```bash
git pull
poetry install
```

This pulls the latest changes and installs any new dependencies.

---

> 💬 If you encounter any installation issues, please open a GitHub issue here:  
> [https://github.com/GdKent/CHAL/issues](https://github.com/GdKent/C-HAL/issues)

