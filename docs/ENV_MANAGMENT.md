# DocRAG — Environment‑Management Guide

*Last updated — 27 May 2025*

---

## 1  Project details

| Item                    | Value                                                                     |
| ----------------------- | ------------------------------------------------------------------------- |
| **Python version**      |  3.12 or newer                                                            |
| **Platforms we target** | • Laptop/CI **CPU** env  <br>• Rental **GPU** box (CUDA 12.4)             |
| **Primary GPU images**  | A40 · RTX A6000 · RTX 4090 (driver ≥ 550)                                 |
| **Env managers**        | *Conda* (for compiled/CUDA libs) + *Poetry* (for pure‑Python & lock‑file) |

### Prerequisites

1. **Conda / Mamba** installed ([https://docs.conda.io](https://docs.conda.io)).
2. **Poetry** installed ([https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)).

---

## 2  Environment‑related files

### Environment-Related Files

| **File**              | **What it defines**                                                                                                                                       |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `environment-cpu.yml` | Defines the Conda environment `docrag-cpu`, used for development and testing on CPU machines. Installs CPU-only versions of heavy libraries like PyTorch (`pytorch=2.6.0`) and Faiss (`faiss-cpu=1.11.0`), along with notebook tools. |
| `environment-gpu.yml` | Defines the Conda environment `docrag-gpu`, targeting GPU machines with CUDA 12.4. Installs `torch==2.6.0+cu124`, `faiss-gpu-cuvs=1.11.0`, `pytorch-cuda=12.4`, and other CUDA-linked libraries. Uses both Conda and pip to install GPU-compatible wheels. |
| `pyproject.toml`      | Defines the project metadata, Python version (`>=3.12`), and all **pure-Python** runtime and dev dependencies. Poetry uses this to manage the environment's Python packages. Conda is not involved here. |
| `poetry.lock`         | Stores the exact resolved versions of every package listed in `pyproject.toml`. Ensures consistent environments across machines. Never edit this file manually—Poetry generates it. |


You probably won't need to interact with these files directly but it's useful to know what they contain. Refer to the [Poetry 'pyproject.toml' documentation](https://python-poetry.org/docs/pyproject/) for more information on the Poetry related files, and the [Conda environment management documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for the Conda related files.

---

## 3  Getting started

### 3.1 First-time setup (run once after cloning)

#### On a CPU-only laptop or CI runner

```bash
# 1. Create the Conda environment from the CPU config
conda env create -f environment_cpu.yml

# 2. Activate the environment
conda activate docrag-cpu

# 3. Install pure-Python dependencies using Poetry
poetry install --with dev

```

#### On a GPU workstation (CUDA 12.4)
```bash
# 1. Create the Conda environment from the GPU config
conda env create -f environment_gpu.yml

# 2. Activate the environment
conda activate docrag-gpu

# 3. Tell Poetry to install into the active Conda env instead of a .venv
poetry config virtualenvs.create false

# 4. Install all pure-Python dependencies
poetry install --with dev
```

### 3.2 Syncing after a `git pull`

After pulling from `master` or another branch, use the following to make sure your environment is fully up to date.

####  Pure-Python updates (Poetry changes only)

```bash
# Must be in the correct Conda env
poetry install --with dev
```

####  Compiled or CUDA packages changed (YAML file updated)

```bash
# CPU machine
conda env update -n docrag-cpu -f environment_cpu.yml --prune

# GPU machine
conda env update -n docrag-gpu -f environment_gpu.yml --prune
```

Then:

```bash
poetry install --with dev
```

>  Always re-run `poetry install` after pulling to apply lock file changes.

---

## 4  Installing a new package

* **Compiled C/CUDA libraries** (they need a compiler or specific CUDA minor)  → **Conda**.
* **Pure‑Python or wheel‑only libraries**  → **Poetry**.


### 4.1  Install **with Conda** (compiled/GPU lib)

```bash
conda activate docrag-gpu                   # or docrag-cpu
conda install <package-name>
conda env export --name docrag-gpu --no-builds | grep -v '^prefix:' > environment_gpu.yml
git add environment-gpu.yml
git commit -m "Added <package-name> <version> to env using Conda."
```

**Always refer to the library or package installation guide for the exact installation command. Some packages require specifying a channel.**

### 4.2  Install **with Poetry** (pure‑Python lib)

```bash
conda activate docrag-cpu                   # ensure env active
poetry add <package-name>
poetry lock
git add pyproject.toml poetry.lock
git commit -m "Added <package-name> <version> to env using Poetry."
```

>  Installing the latest version of a package may not always be compatible, use `<package-name>@<version>` or `<package-name>=<version>` for more control.

---

## 5  Examples of common packages and how to install them

| Category     | Package examples                                                                | Install via            |
| ------------ | ------------------------------------------------------------------------------- | ---------------------- |
| Heavy GPU ML | torch, torchvision, pytorch‑cuda, faiss‑gpu‑cuVS, flash‑attention, bitsandbytes | **Conda**              |
| CPU math     | numpy, scipy, scikit‑learn, pandas                                              | Conda preferred (perf) |
| Text / NLP   | transformers, datasets, peft, accelerate                                        | **Poetry**             |
| Web / CLI    | fastapi, uvicorn, typer, rich                                                   | **Poetry**             |
| Dev tools    | pytest, ruff, black, mypy, pre‑commit                                           | **Poetry** (dev group) |
