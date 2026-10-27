.PHONY: setup lint format typecheck test check up down migrate prepare train bootstrap replay \
        allocate monitor retrain parity loadtest figures reset demo clean monitoring-up monitoring-down \
        alerting-experiment rules-check

RETRAIN_DAY ?= 165
LOAD_USERS ?= 32
LOAD_TIME ?= 60s
LOAD_TAG ?= u$(LOAD_USERS)

setup:  ## install deps (on macOS without Homebrew libomp, patch LightGBM to use scikit-learn's copy)
	uv sync
	@if [ "$$(uname)" = "Darwin" ]; then ./scripts/macos_libomp.sh; fi
	uv run pre-commit install

lint:
	uv run ruff check src tests loadtest scripts
	uv run ruff format --check src tests loadtest scripts

format:
	uv run ruff check --fix src tests loadtest scripts
	uv run ruff format src tests loadtest scripts

typecheck:
	uv run mypy

test:
	uv run pytest --cov=fraudrt --cov-report=term-missing:skip-covered

check: lint typecheck test

up:  ## whole stack: Postgres, Redis, Redpanda, MLflow, migrations, API, stream worker
	docker compose up -d --build

down:
	docker compose --profile monitoring down

monitoring-up:  ## opt-in: Prometheus, Alertmanager, Grafana (http://localhost:3000), issue receiver
	mkdir -p work/issues
	docker compose --profile monitoring up -d --build

monitoring-down:
	docker compose --profile monitoring stop prometheus alertmanager grafana alert-receiver

rules-check:  ## promtool/amtool via Docker: rule syntax + unit tests
	docker run --rm --entrypoint promtool -v $(PWD)/ops/prometheus:/p:ro prom/prometheus:v2.55.1 check rules /p/alerts.yml
	docker run --rm --entrypoint promtool -v $(PWD)/ops/prometheus:/p:ro prom/prometheus:v2.55.1 test rules /p/alerts_test.yml
	docker run --rm --entrypoint amtool -v $(PWD)/ops/alertmanager:/a:ro prom/alertmanager:v0.27.0 check-config /a/alertmanager.yml

migrate:
	uv run alembic upgrade head

prepare:  ## Kaggle CSVs -> cached offline feature table
	uv run fraudrt prepare

train:  ## LightGBM -> MLflow -> registry alias "champion" (API picks it up on its own)
	uv run fraudrt train

bootstrap:  ## warm Redis with the history before the replay period
	uv run fraudrt bootstrap

replay:  ## publish the hold-out period to Redpanda; waits until the worker has drained it
	uv run fraudrt replay $(if $(SPEEDUP),--speedup $(SPEEDUP),)

parity:  ## features served online == offline features; labels never stored before scoring
	uv run python scripts/check_parity_live.py
	uv run python scripts/check_label_gate.py

allocate:
	uv run fraudrt allocate

monitor:
	uv run fraudrt monitor

retrain:  ## retrain using only labels known at day $(RETRAIN_DAY); registers a challenger
	uv run fraudrt retrain $(RETRAIN_DAY)

loadtest:  ## Locust against /score; run after the replay (synthetic rows are purged from Postgres afterwards)
	uv run locust -f loadtest/locustfile.py --headless -u $(LOAD_USERS) -r $(LOAD_USERS) \
	  -t $(LOAD_TIME) --host http://localhost:8000 --csv work/results/locust_$(LOAD_TAG) --only-summary
	uv run python scripts/summarise_loadtest.py $(LOAD_TAG)
	uv run fraudrt purge-loadtest

figures:
	uv run python scripts/make_figures.py

reset:  ## empty Postgres tables, Redis keys and Kafka topics
	docker compose stop worker
	uv run fraudrt reset
	docker compose start worker

demo: up train bootstrap replay parity allocate monitor retrain figures  ## end to end

clean:
	rm -rf work .pytest_cache .mypy_cache .ruff_cache .coverage
