#!/bin/bash
set -euo pipefail
source venv/bin/activate
VERSION=$(python3 -m setuptools_scm)
echo docker build --build-arg="APP_VERSION=$VERSION" -t differential_verification .
docker build --build-arg="APP_VERSION=$VERSION" -t differential_verification .