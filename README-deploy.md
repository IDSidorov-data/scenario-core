## Build and run with Docker

# Build image
docker build -t scenario-core:latest .

# Run container (exposes 8501)
docker run -e PORT=8501 -p 8501:8501 scenario-core:latest

# Or with docker-compose
docker compose up --build

## Deploy to Heroku
# 1. Create app on Heroku
# 2. Set buildpack and push
heroku create my-scenario-app
# Push to Heroku remote
git push heroku main
# Heroku will use the Procfile

# Notes:
# - Store secrets (SENTRY_DSN, AI keys, SUPABASE keys) in platform env vars, not in repo.
# - For Next.js frontend, prefer Vercel; Docker is optional for frontend but not required.