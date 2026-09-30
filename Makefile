# UniMate synchronous HTTP API (testing) — build and run.
#
#   make build-base             # deps only; rebuild only when requirements change
#   make build weights          # API image, then seed ./unimate-weights once (~1.3 GB)
#   make dev                    # run it via compose on :$(PORT), weights mounted
#   make serve                  # no Docker: uvicorn from the current env, same weights
#   make request REQ=req.json   # POST a request to a running API
.PHONY: help build-base push-base build weights run dev down logs serve request

DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
IMAGE_USER ?= sybiote
BASE_IMAGE := $(IMAGE_USER)/unimate-base:latest
HTTP_IMAGE := unimate-http:latest
# 8010, not 8000: the local Kimodo container holds 8000.
PORT ?= 8010
REQ ?= request.json
WEIGHTS ?= $(CURDIR)/unimate-weights

help:
	@echo "Targets:"
	@echo "  make build-base   Build unimate-base (deps only; weights are mounted)"
	@echo "  make push-base    Push the base to Docker Hub"
	@echo "  make build        Build $(HTTP_IMAGE) on the local base"
	@echo "  make weights      Seed $(WEIGHTS) once (pinned checkpoint, EMA-only, + flan-t5-base)"
	@echo "  make run          Run it with the GPU on :$(PORT)"
	@echo "  make dev          docker compose up --build (detached) on :$(PORT)"
	@echo "  make down / logs  Stop the compose service / follow its logs"
	@echo "  make serve        Run api.py without Docker, on the same $(WEIGHTS)"
	@echo "  make request      POST \$$REQ to localhost:$(PORT)/v1/motion"

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t unimate-base:latest -t $(BASE_IMAGE) .

push-base:
	docker push $(BASE_IMAGE)

build:
	$(DOCKER_BUILD) --build-arg BASE_IMAGE=unimate-base:latest \
		-f Dockerfile.http -t $(HTTP_IMAGE) .

weights:
	mkdir -p $(WEIGHTS)
	# Online explicitly (the serving env is offline) and as you, so the files are yours.
	docker run --rm --user $$(id -u):$$(id -g) -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -e HF_HOME=/tmp/hf \
		-v $(WEIGHTS):/weights $(HTTP_IMAGE) python scripts/seed_weights.py /weights

run:
	docker run --rm --gpus all -p $(PORT):8000 -v $(WEIGHTS):/weights:ro -e UNIMATE_WEIGHTS_ROOT=/weights $(HTTP_IMAGE)

dev:
	HOST_PORT=$(PORT) docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f unimate-api

serve:
	UNIMATE_WEIGHTS_ROOT=$(WEIGHTS) ./docker-entrypoint.sh uvicorn api:app --host 0.0.0.0 --port $(PORT)

request:
	curl -sS -X POST localhost:$(PORT)/v1/motion -H 'Content-Type: application/json' -d @$(REQ)
