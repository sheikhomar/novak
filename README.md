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

6. Configure Google Drive Remote in DVC using [Google Service Account credentials](https://dvc.org/doc/user-guide/setup-google-drive-remote#using-service-accounts):

   ```bash
   # Download and copy credentials file to .dvc/tmp/gdrive-user-credentials.json
   poetry run dvc remote modify gdrive-remote \
      --local gdrive_service_account_json_file_path .dvc/tmp/gdrive-user-credentials.json
   ```

## Preparing AG News data

```bash
# Download data from the DVC remote
poetry run dvc pull data/input/raw/agnews.zip

# Prepare the raw data
poetry run python -m app.data.prep_agnews \
   --download-url https://corise-mlops.s3.us-west-2.amazonaws.com/project1/agnews.zip \
   --output-dir data/input
```
