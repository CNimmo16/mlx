# MLX course

## Setup

1. Ensure Python is installed on the system
2. Install PDM with `curl -sSL https://pdm-project.org/install-pdm.py | python3 -` (you may need to add to your path after doing this to allow running the `pdm` command, see output of installation script for details)
3. Point PDM to a python interpreter - run `which python` then pass the path you want to use to `pdm use <path-to-python>`. PDM will automatically create a virtual environment in the `.venv` folder
4. Run `source .venv/bin/activate` to activate the virtual environment
5. Run `pdm install` to install dependencies

## Running scripts

Activate virtual environment (if not already activated) using `source .venv/bin/activate)` then run `python <path-to-script>`
