# Deployment (free tiers only)

The service is a stateless container that serves `results/latest_forecasts.json`.
It never fits models at request time, so the smallest free instance is enough.

## Local
```bash
make install
make data forecast        # needs internet access to Yahoo Finance
make serve                # http://localhost:8000/docs
```
No data yet? Serve the clearly-labelled synthetic example:
```bash
VOLTAIL_FORECASTS_FILE=examples/latest_forecasts.synthetic.json make serve
```

## Docker
```bash
docker compose up --build           # API :8000
docker compose run --rm pipeline    # refresh real forecasts into ./results
VOLTAIL_FORECASTS_FILE=/app/results/latest_forecasts.json docker compose up -d api   # serve them
```

## Render (free web service)
1. Push the repo to GitHub (the scheduled `refresh.yml` workflow keeps `results/latest_forecasts.json` current in the repo).
2. On render.com: New → Web Service → connect the repo → Runtime **Docker**, instance type **Free**.
3. Environment variable: `VOLTAIL_FORECASTS_FILE=/app/results/latest_forecasts.json`. Also remove `results` from `.dockerignore` (or add a `COPY results ./results` line) so the committed forecast file is baked into the image.
4. Health check path: `/health`. Turn on auto-deploy so each refresh commit redeploys with new forecasts.

Free instances sleep when idle; the first request after a sleep takes about 30–60 seconds.

## Hugging Face Spaces (Docker Space, free CPU)
1. Create a Space → SDK **Docker**.
2. Push this repo to the Space. Add to the top of the README the YAML header `sdk: docker` and `app_port: 8000`.
3. Same `VOLTAIL_FORECASTS_FILE` / `results` note as for Render.

## Fly.io
`fly launch` detects the Dockerfile; set `internal_port = 8000` and the same
environment variable. Fly's free allowance changes over time, so check their pricing
page first; nothing here requires a paid feature.

## What is *not* deployed
The walk-forward backtest is a batch job (about 10 minutes on 2 CPU cores; longer on smaller
machines). It runs locally or in the `full_backtest.yml` GitHub Action, not on the web
host.
