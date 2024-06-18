#!/bin/bash
set -euo pipefail
docker build -t autodiver .
docker run -v .:/mnt  --rm -it autodiver /bin/bash -c 'cd /mnt; exec bash'