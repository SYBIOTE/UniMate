# UniMate RunPod queue worker — build and push.
#
#   make build-base push-base   # deps + weights; rebuild only when those change
#   make build push             # queue worker on top of the base (code changes)
#   make test                   # run handler.py locally on a job JSON
.PHONY: help build-base push-base build push test

DOCKER_BUILD := DOCKER_BUILDKIT=1 docker build
IMAGE_USER ?= sybiote
TAG ?= $(shell git rev-parse --short HEAD)
BASE_IMAGE := $(IMAGE_USER)/unimate-base:latest
QUEUE_IMAGE := $(IMAGE_USER)/unimate-queue
JOB ?= job.json

help:
	@echo "Targets:"
	@echo "  make build-base   Build unimate-base (deps + baked weights)"
	@echo "  make push-base    Push the base to Docker Hub"
	@echo "  make build        Build unimate-queue:$(TAG) on the local base"
	@echo "  make push         Push unimate-queue:$(TAG) (set it as the endpoint image)"
	@echo "  make test JOB=... Run the handler locally on a job file (needs UNIMATE_EXP_DIR)"

build-base:
	$(DOCKER_BUILD) -f Dockerfile.base -t unimate-base:latest -t $(BASE_IMAGE) .

push-base:
	docker push $(BASE_IMAGE)

build:
	$(DOCKER_BUILD) --build-arg BASE_IMAGE=unimate-base:latest \
		-f Dockerfile -t $(QUEUE_IMAGE):$(TAG) -t $(QUEUE_IMAGE):latest .

push:
	docker push $(QUEUE_IMAGE):$(TAG)
	docker push $(QUEUE_IMAGE):latest

test:
	python handler.py --test_input "$$(cat $(JOB))"
