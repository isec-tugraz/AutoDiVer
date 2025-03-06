#!/bin/bash
set -euo pipefail

# chdir to script location
cd "$(dirname "$0")"

docker build -t autodiver .
docker run -v "$PWD:/mnt"  --rm -it autodiver /bin/bash -c 'cd /mnt; exec bash'
