#!/usr/bin/env bash

# Host directory (where this script is run from)
HOST_DIR="$(pwd)"
CONTAINER_DIR="/opt/app/working"

# Path to the script you want to run (relative to host OpenLimbTT)
SCRIPT_PATH="$CONTAINER_DIR/GenerateRandomLimbs.py"

# Optional: Pass extra args to the script from the host command line
EXTRA_ARGS="$@"

# Run it inside the Docker container
docker run --rm -it \
  -v "$HOST_DIR":"$CONTAINER_DIR" \
  -w "$CONTAINER_DIR" \
  openlimbtt-env \
  python "$SCRIPT_PATH" $EXTRA_ARGS
