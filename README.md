# Dockerized ML Model (Iris)

Tiny REST API that serves a scikit-learn model via Flask & Docker.

## Run (Docker)
```bash
docker build -t docker-ml-model .
docker run -p 5000:5000 docker-ml-model
