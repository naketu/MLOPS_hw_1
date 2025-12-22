IMAGE ?= your_dockerhub_username/mlops_hw
TAG ?= latest

.PHONY: docker-push test lint

docker-push:
	docker build -t $(IMAGE):$(TAG) .
	docker push $(IMAGE):$(TAG)

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .
