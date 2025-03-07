#!/bin/bash
# docker_visualize.sh
#
# Usage:
#   bash docker_visualize.sh <trained_model_location> <scene_location> <port> <ip> [additional host_render_server.py flags]
#
# This script mounts the model and scene directories into the container at /data/trained_model and /data/scene,
# exposes the specified port on the given IP, and then executes:
#   python host_render_server.py -m /data/trained_model -s /data/scene --port <port> --ip <ip> [extra flags]

if [ "$#" -lt 4 ]; then
  echo "Usage: $0 <trained_model_location> <scene_location> <port> <ip> [additional host_render_server.py flags]"
  exit 1
fi

TRAINED_MODEL_LOCATION="$1"
SCENE_LOCATION="$2"
PORT="${3:-6009}"
IP="${4:-127.0.0.1}"
shift 4

docker run --rm --gpus all -it \
  -v /tmp/NVIDIA:/tmp/NVIDIA \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v "$TRAINED_MODEL_LOCATION":/data/trained_model \
  -v "$SCENE_LOCATION":/data/scene \
  -p "$IP:$PORT:$PORT" \
  -v "$(pwd)":/ever_training2 \
  ever \
  bash -c "source activate ever && cd /ever_training2 && rm -r ever && cp -r /ever_training/ever . && python host_render_server.py -m /data/trained_model -s /data/scene --port $PORT --ip $IP $*"

