# Human Promoter Atlas — common dev / build / deploy tasks
# Run with `make <target>` from the repo root.

PY            := python
PORT          := 8501
DB            := data/canonical_promoter.duckdb
COMPOSE       := docker compose -f deploy/docker-compose.yml

.PHONY: help db dev stop image up down logs restart push clean

help:
	@echo "Human Promoter Atlas — make targets"
	@echo "  make db        — (re)build data/canonical_promoter.duckdb from analyses/"
	@echo "  make dev       — run streamlit locally on port $(PORT)"
	@echo "  make stop      — kill any local streamlit on port $(PORT)"
	@echo "  make image     — docker build the app image"
	@echo "  make up        — docker compose up -d --build (Caddy + app)"
	@echo "  make down      — docker compose down"
	@echo "  make restart   — docker compose restart app"
	@echo "  make logs      — tail compose logs"
	@echo "  make clean     — remove built duckdb + aggregate parquets"

# Build the data layer from the upstream analyses/ outputs
db:
	$(PY) data/build_app_db.py
	@echo "  wrote $(DB) ($$(du -h $(DB) | cut -f1))"

# Local dev server
dev:
	@$(MAKE) stop >/dev/null 2>&1 || true
	streamlit run app/streamlit_app.py --server.port=$(PORT) --server.headless=true

stop:
	-pkill -f 'streamlit_app.py' 2>/dev/null
	@sleep 1

# Docker
image:
	docker build -t human-promoter-atlas:latest -f app/Dockerfile .

up:
	$(COMPOSE) up -d --build

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart app

logs:
	$(COMPOSE) logs -f --tail=100

clean:
	rm -f $(DB)
	rm -rf data/aggregate
	@echo "removed built data artifacts"
