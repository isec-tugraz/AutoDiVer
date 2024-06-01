#!/bin/bash
set -euo pipefail
docker build -t differential_verification .
docker run -v .:/mnt  --rm -it differential_verification /bin/bash -c 'cd /mnt; exec bash'