# Deployment guide — Kaya RAG

This file shows quick ways to deploy the app locally (Docker), to a Linux VM, or to a container hosting provider.

Prereqs (local)
- Docker + docker-compose
- Optional: Node (for local frontend dev) and Python 3.12 (for backend venv)

1) Local (recommended quick start) — Docker Compose

- Build and start both services:

```bash
docker-compose build --pull
docker-compose up -d
```

- The frontend will be at http://localhost:5173 and the backend at http://localhost:8000
- Check health: curl http://localhost:8000/health
- Persistent data: `./backend/chroma_db` is mounted from the host.

Notes:
- Ensure `backend/.env` contains your NVIDIA API key and model selection before starting.
- The first run may download embedding models (~80MB) and take longer.

2) Deploy to a cloud VM (Ubuntu 22.04 / Debian)

- Copy the repo to the VM, install Docker and docker-compose, then run the same `docker-compose` commands above.
- Alternatively, run the backend in a Python venv and the frontend as a static site served by nginx.

3) Container hosting (AWS ECS, GCP Cloud Run, Fly.io, Render)

- Build and push the two images to a registry (Docker Hub / ECR / GCR):

```bash
docker build -t <registry>/kaya-rag-backend:latest -f backend/Dockerfile .
docker push <registry>/kaya-rag-backend:latest

docker build -t <registry>/kaya-rag-frontend:latest -f frontend/Dockerfile .
docker push <registry>/kaya-rag-frontend:latest
```

- For hosted platforms that run a single container (Cloud Run, Fly), it's easiest to host the frontend as a static site (Netlify / Vercel / S3+CloudFront) and deploy the backend container to the provider.

Verification

- Backend logs: docker-compose logs -f backend
- Frontend logs: docker-compose logs -f frontend
- Use the sample flow: ingest `backend/data/sample.txt` via the UI and then ask a question.

Security

- Don't commit `backend/.env` to git. Rotate API keys after use.
