# Novak

News categorization.

## Development Setup

1. Install [pyenv](https://github.com/pyenv/pyenv):

   ```bash
   curl https://pyenv.run | bash
   ```

2. Install and activate the Python version:

   ```bash
   pyenv install
   ```

3. Install [Poetry](https://python-poetry.org/docs/).
   ```bash
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
   source $HOME/.poetry/env
   ```

4. Install dependencies

   ```bash
   poetry install
   poetry env info # Show virtualenv information
   ```

5. Install the pre-commit Git hooks:

   ```bash
   poetry run pre-commit install
   ```
