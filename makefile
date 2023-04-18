IMAGE_NAME = "pss-tf"

docker_build:
	docker build -t $(IMAGE_NAME) .

docker_run_detech:
	docker run -d $(IMAGE_NAME)

docker_run:
	docker run $(IMAGE_NAME)

docker_exec:
	docker exec -it $(IMAGE_NAME) /bin/bash

docker_clean::
	docker rm $(docker ps -a -q)
