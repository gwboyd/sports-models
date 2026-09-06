SHELL := /bin/zsh

.PHONY: help backend frontend frontend-local frontend-sam sync-cfb-teams sync-nfl-teams sync-football-teams backtest-expected-points sam-build sam-invoke-health sam-api sam-deploy sam-register-releases sam-finalize-release-files

YEAR ?= $(shell date +%Y)

help:
	@echo "Available targets:"
	@echo "  make backend            Run FastAPI locally on 127.0.0.1:3000"
	@echo "  make frontend           Run Next.js using frontend/.env as-is"
	@echo "  make frontend-local     Run Next.js against direct local backend (127.0.0.1:3000)"
	@echo "  make frontend-sam       Run Next.js against local SAM API (127.0.0.1:3001)"
	@echo "  make sync-cfb-teams     Refresh CFB team metadata/logos (YEAR=current year)"
	@echo "  make sync-nfl-teams     Refresh NFL metadata/logos from nflverse"
	@echo "  make sync-football-teams Refresh both football team catalogs"
	@echo "  make backtest-expected-points Compare an expected-points candidate with a baseline"
	@echo "  make sam-build          Build the Lambda image/artifacts with SAM"
	@echo "  make sam-invoke-health  Invoke the API Lambda with the sample health event"
	@echo "  make sam-api            Run the SAM local API on 127.0.0.1:3001"
	@echo "  make sam-deploy         Interactively version and deploy the stack to us-east-1"
	@echo "  make sam-register-releases Retry release registration after a successful SAM deploy"
	@echo "  make sam-finalize-release-files Archive/reset drafts from the last deployment plan"

backend:
	source .venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 3000 --reload

frontend:
	cd frontend && npm run dev

frontend-local:
	cd frontend && ENDPOINT=http://127.0.0.1:3000 npm run dev

frontend-sam:
	cd frontend && ENDPOINT=http://127.0.0.1:3001 npm run dev

sync-cfb-teams:
	set -a && source .env && set +a && cd frontend && npm run sync:cfb-teams -- --year $(YEAR)

sync-nfl-teams:
	cd frontend && npm run sync:nfl-teams

sync-football-teams:
	set -a && source .env && set +a && cd frontend && npm run sync:football-teams -- --year $(YEAR)

LEAGUE ?= nfl
PROFILE ?= standard
BASELINE ?= deployed
CANDIDATE ?= working-tree
FRAME ?= .backtests/expected_points/frames/$(LEAGUE)/latest.parquet
BASELINE_FRAME ?=
CANDIDATE_FRAME ?=
SEASONS ?=
CADENCE ?=
BOOTSTRAP_SAMPLES ?= 2000
OUTPUT_DIR ?=
NO_CACHE ?=
CACHE_WORKING_TREE ?=
BACKTEST_FRAME_ARGS = --frame $(FRAME)
ifneq ($(strip $(BASELINE_FRAME)),)
BACKTEST_FRAME_ARGS += --baseline-frame $(BASELINE_FRAME)
endif
ifneq ($(strip $(CANDIDATE_FRAME)),)
BACKTEST_FRAME_ARGS += --candidate-frame $(CANDIDATE_FRAME)
endif
BACKTEST_OPTIONAL_ARGS = --bootstrap-samples $(BOOTSTRAP_SAMPLES)
BACKTEST_TRUE_VALUES = 1 true TRUE yes YES
ifneq ($(strip $(SEASONS)),)
BACKTEST_OPTIONAL_ARGS += --seasons $(SEASONS)
endif
ifneq ($(strip $(CADENCE)),)
BACKTEST_OPTIONAL_ARGS += --cadence $(CADENCE)
endif
ifneq ($(strip $(OUTPUT_DIR)),)
BACKTEST_OPTIONAL_ARGS += --output-dir $(OUTPUT_DIR)
endif
ifneq ($(filter $(BACKTEST_TRUE_VALUES),$(strip $(NO_CACHE))),)
BACKTEST_OPTIONAL_ARGS += --no-cache
endif
ifneq ($(filter $(BACKTEST_TRUE_VALUES),$(strip $(CACHE_WORKING_TREE))),)
BACKTEST_OPTIONAL_ARGS += --cache-working-tree
endif

backtest-expected-points:
	set -a && { [[ ! -f .env ]] || source .env; } && set +a && \
	.venv/bin/python scripts/backtest_expected_points.py compare \
		--league $(LEAGUE) \
		--profile $(PROFILE) \
		--baseline $(BASELINE) \
		--candidate $(CANDIDATE) \
		$(BACKTEST_FRAME_ARGS) \
		$(BACKTEST_OPTIONAL_ARGS)

sam-build:
	sam build

sam-invoke-health:
	set -a && source .env && set +a && \
	sam local invoke ApiLambdaFunction \
		-e events/get-health-event.json \
		--parameter-overrides \
			Localhost=True \
			EnvironmentName=LOCAL \
			AdminApiKey="$$ADMIN_API_KEY" \
			FrontEndApiKey="$$FRONT_END_API_KEY" \
			ReadApiKey="$$READ_API_KEY" \
			NbaApiKey="$$NBA_API_KEY" \
			AwsApiKey="$$AWS_API_KEY" \
			CfbdApiKey="$$CFBD_API_KEY" \
			SupabaseDbUrl="$$SUPABASE_DB_URL" \
			SupabaseSchema="$$SUPABASE_SCHEMA"

sam-api:
	set -a && source .env && set +a && \
	sam local start-api \
		--parameter-overrides \
			Localhost=True \
			EnvironmentName=LOCAL \
			AdminApiKey="$$ADMIN_API_KEY" \
			FrontEndApiKey="$$FRONT_END_API_KEY" \
			ReadApiKey="$$READ_API_KEY" \
			NbaApiKey="$$NBA_API_KEY" \
			AwsApiKey="$$AWS_API_KEY" \
			CfbdApiKey="$$CFBD_API_KEY" \
			SupabaseDbUrl="$$SUPABASE_DB_URL" \
			SupabaseSchema="$$SUPABASE_SCHEMA"

sam-deploy:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py

sam-register-releases:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py --register-only

sam-finalize-release-files:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py --finalize-only
