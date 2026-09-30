# UniMate -- RunPod Serverless (queue) worker.
#
# RunPod's Git build builds this file from the runpod-queue branch; the heavy
# layer (deps + weights) comes from Dockerfile.base on Docker Hub.
#
#   docker build -f Dockerfile.base -t unimate-base:latest .
#   docker build --build-arg BASE_IMAGE=unimate-base:latest -t unimate-queue:latest .
#
# No port is exposed and there is no HTTP healthcheck: the worker pulls jobs off
# the queue rather than serving requests, so readiness is the SDK connecting,
# not a socket accepting.

ARG BASE_IMAGE=docker.io/sybiote/unimate-base:latest
FROM ${BASE_IMAGE}

COPY requirements-queue.txt .
RUN python -m pip install -r requirements-queue.txt

# The base image bakes unimate/ as of whenever it was last built (it needs the
# code to slim the checkpoint). Copy it again so this branch's edits reach the
# worker without rebuilding the base.
COPY unimate/ unimate/
COPY data_process/ data_process/
COPY handler.py handler.py

CMD ["python", "-u", "handler.py"]
