#!/usr/bin/env bash
set -e
set -x

SRC=${1:-"src/zenml tests"}

# mypy src/zenml
flake8 $SRC
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place $SRC --exclude=__init__.py,legacy/* --check
isort $SRC scripts --check-only
black $SRC  --check
interrogate $SRC -c pyproject.toml