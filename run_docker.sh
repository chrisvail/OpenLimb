#!/usr/bin/env bash

# Absolute path to the current directory (where OpenLimbTT is assumed to be)
HOST_DIR="$(pwd)"
CONTAINER_DIR="/opt/app/working"

# Run the Docker container interactively, mounting the OpenLimbTT directory
docker run --rm -it \
  -v "$HOST_DIR":"$CONTAINER_DIR" \
  -w "$CONTAINER_DIR" \
  openlimbtt-env \
  bash
