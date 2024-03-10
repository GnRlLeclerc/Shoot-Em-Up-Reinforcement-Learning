# INF581 - Reinforcement Learning on a ShootEm Up

[![Python Version](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)

This repository contains the code for a simple ShootEm Up game, and the implementation of various reinforcement
learning agents in order to play it.

Setups for VSCode and Pycharm are included. Instructions on how to use the project build tools in command line can be 
accessed [here](#build-tools).

## Project structure

```bash
├── .idea     # PyCharm configuration
├── .vscode   # VSCode configuration
│
├── game      # Main source directory
│   ├── backend            # Game backend logic
│   ├── frontend           # Game frontend display using PyGame
│   ├── rl_agents          # Reinforcement Learning agent definition
│   └── rl_environment     # Reinforcement Learning game environment wrapper
│
├── notebook  # Demo Jupyter notebook
│
└── scripts   # Ready-to-run scripts
    ├── example.py         # Example script
    ├── run_game.py        # Run the game in order to play it yourself
    └── setup.py           # Python PATH setup to import at the top of each script
```

## Quickstart

### Poetry

This project manages its dependencies and python version with [poetry](https://python-poetry.org/). You can find installation instructions [here](https://python-poetry.org/docs/#installing-with-pipx).

Install the dependencies with poetry:
You must first install `Python 3.11` on your machine (preferably using a manager like conda, pyenv...)

```bash
poetry install
```

If this command fails, (install is stuck on _'Pending...'_) you can run the following command before retrying :

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```

The issue is described [here](https://github.com/python-poetry/poetry/issues/6668). Otherwise, you can install the dependencies manually from the `pyproject.toml` file.

This will create a virtual environment in the `.venv` root folder. You can activate it with:

```bash
poetry shell
```

Add, remove and update dependencies with the following commands

```bash
poetry add <package>
poetry remove <package>
poetry update
```

You can then run scripts with:
```bash
python scripts/example.py  # Replace with any script
```

Launch the game with:
```bash
python scripts/run_game.py
```

You can control the player with the arrow keys. It will look in the direction of your mouse.

### Build tools
This project uses [`Black`](https://pypi.org/project/black/) for code formatting, [`PyLint`](https://pypi.org/project/pylint/) 
for linting and [`isort`](https://pypi.org/project/isort/) for import statements sorting.
Checks are run using a [`Gitlab-CI`](./.gitlab-ci.yml) pipeline in order to ensure that the code is properly formatted, linted and sorted.

Add the recommended extensions from the `.vscode/extensions.json` file to your VSCode workspace.
This project is based on the `Python` extension from Microsoft. It uses `Black` for formatting and `Pylint` for linting.

Use the following shortcuts for everything formatting and linting related:
These shortcuts are written in the `[tool.poe.tasks]` section of the [`pyproject.toml`](./pyproject.toml) file.

Format the code:

```bash
poe format-write
```

Check that the code has been formatted (used only in gitlab-ci checks):

```bash
poe format-check
```

Lint the code / check that there are no errors or warnings:

```bash
poe lint
```

Sort the imports:

```bash
poe sort-imports
```

Check that the imports are correctly sorted:

```bash
poe sort-check
```
