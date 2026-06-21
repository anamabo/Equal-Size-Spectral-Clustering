install-hooks:
	poetry run pre-commit install

format:
	poetry run ruff format .
	poetry run black .
