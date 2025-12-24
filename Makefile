SHELL := /bin/bash

# Use like:
# make docker-push DOCKERHUB_USER=nikitaaksenov IMAGE_NAME=mlops_hw TAG=v0.1.0
DOCKERHUB_USER ?=
IMAGE_NAME ?= mlops_hw
IMAGE ?= $(DOCKERHUB_USER)/$(IMAGE_NAME)
TAG ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)

.PHONY: docker-push test lint

docker-push:
	@if [[ -z "$(DOCKERHUB_USER)" ]]; then \
		echo "ERROR: DOCKERHUB_USER is empty. Run: make docker-push DOCKERHUB_USER=<your_user>"; \
		exit 1; \
	fi
	@docker info >/dev/null 2>&1 || (echo "ERROR: Docker daemon is not running" && exit 1)
	@docker login >/dev/null 2>&1 || (echo "ERROR: Not logged in to Docker Hub. Run: docker login" && exit 1)
	docker build -t $(IMAGE):$(TAG) -t $(IMAGE):latest .
	docker push $(IMAGE):$(TAG)
	docker push $(IMAGE):latest

test:
	poetry run pytest -q

lint:
	poetry run ruff check .
	poetry run ruff format --check .
