# UniMate synchronous HTTP API (testing) — build and run.
#
#   make build-base             # deps + weights; rebuild only when those change
#   make build run              # API image on top of the base, served on :$(PORT)
#   make serve                  # no Docker: uvicorn from the current env
#   make request REQ=req.json   # POST a request to a running API
.PHONY: help build-base push-base build run serve request

DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
IMAGE_USER ?= sybiote
BASE_IMAGE := $(IMAGE_USER)/unimate-base:latest
HTTP_IMAGE := unimate-http:latest
PORT ?= 8000
REQ ?= request.json

help:
	@echo "Targets:"
	@echo "  make build-base   Build unimate-base (deps + baked weights)"
	@echo "  make push-base    Push the base to Docker Hub"
	@echo "  make build        Build $(HTTP_IMAGE) on the local base"
	@echo "  make run          Run it with the GPU on :$(PORT)"
	@echo "  make serve        Run api.py without Docker (needs UNIMATE_EXP_DIR)"
	@echo "  make request      POST \$$REQ to localhost:$(PORT)/v1/motion"

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t unimate-base:latest -t $(BASE_IMAGE) .

push-base:
	docker push $(BASE_IMAGE)

build:
	$(DOCKER_BUILD) --build-arg BASE_IMAGE=unimate-base:latest \
		-f Dockerfile.http -t $(HTTP_IMAGE) .

run:
	docker run --rm --gpus all -p $(PORT):8000 $(HTTP_IMAGE)

serve:
	uvicorn api:app --host 0.0.0.0 --port $(PORT)

request:
	curl -sS -X POST localhost:$(PORT)/v1/motion -H 'Content-Type: application/json' -d @$(REQ)
