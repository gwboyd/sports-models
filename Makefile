SHELL := /bin/zsh

.PHONY: help backend frontend frontend-local frontend-sam sync-cfb-teams sync-nfl-teams sync-football-teams backtest-expected-points replay-expected-points-locks sam-build sam-invoke-health sam-api prepare-model-release sam-deploy sam-register-releases

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
	@echo "  make backtest-expected-points Compare one league; LEAGUES=nfl,cfb starts both for convenience"
	@echo "  make replay-expected-points-locks Screen Lock methods from a frozen score-run cache"
	@echo "  make sam-build          Build the Lambda image/artifacts with SAM"
	@echo "  make sam-invoke-health  Invoke the API Lambda with the sample health event"
	@echo "  make sam-api            Run the SAM local API on 127.0.0.1:3001"
	@echo "  make prepare-model-release Choose versions and prepare release notes before commit/merge"
	@echo "  make sam-deploy         Confirm and deploy committed versions from clean main"
	@echo "  make sam-register-releases Retry release registration after a successful SAM deploy"

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
LEAGUES ?=
PROFILE ?= standard
BASELINE ?= deployed
CANDIDATE ?= working-tree
FRAME ?=
BASELINE_FRAME ?=
CANDIDATE_FRAME ?=
SEASONS ?=
CADENCE ?=
BOOTSTRAP_SAMPLES ?= 2000
OUTPUT_DIR ?=
NO_CACHE ?=
CACHE_WORKING_TREE ?=
LOCK_SOURCE ?= deployed
LOCK_EXPECTED_VERSION ?=
LOCK_SCREEN_CADENCE ?=
LOCK_MIN_PROBABILITY ?= 0.55
LOCK_MAX_PER_WEEK ?= 5
LOCK_EXTRA_MIN_PROBABILITY ?=
THROUGH_SEASON ?=
PREFLIGHT_ONLY ?=
CURRENT_YEAR ?=
CURRENT_WEEK ?=
BACKTEST_FRAME_ARGS =
ifneq ($(strip $(FRAME)),)
BACKTEST_FRAME_ARGS += --frame "$(FRAME)"
endif
ifneq ($(strip $(BASELINE_FRAME)),)
BACKTEST_FRAME_ARGS += --baseline-frame "$(BASELINE_FRAME)"
endif
ifneq ($(strip $(CANDIDATE_FRAME)),)
BACKTEST_FRAME_ARGS += --candidate-frame "$(CANDIDATE_FRAME)"
endif
BACKTEST_OPTIONAL_ARGS = --bootstrap-samples $(BOOTSTRAP_SAMPLES)
ifneq ($(strip $(CURRENT_YEAR)),)
BACKTEST_OPTIONAL_ARGS += --current-year $(CURRENT_YEAR)
endif
ifneq ($(strip $(CURRENT_WEEK)),)
BACKTEST_OPTIONAL_ARGS += --current-week $(CURRENT_WEEK)
endif
BACKTEST_TRUE_VALUES = 1 true TRUE yes YES
ifneq ($(strip $(THROUGH_SEASON)),)
BACKTEST_OPTIONAL_ARGS += --through-season $(THROUGH_SEASON)
endif
ifneq ($(filter $(BACKTEST_TRUE_VALUES),$(strip $(PREFLIGHT_ONLY))),)
BACKTEST_OPTIONAL_ARGS += --preflight-only
endif
ifneq ($(strip $(SEASONS)),)
BACKTEST_OPTIONAL_ARGS += --seasons $(SEASONS)
endif
ifneq ($(strip $(CADENCE)),)
BACKTEST_OPTIONAL_ARGS += --cadence $(CADENCE)
endif
ifneq ($(strip $(OUTPUT_DIR)),)
BACKTEST_OPTIONAL_ARGS += --output-dir "$(OUTPUT_DIR)"
endif
ifneq ($(filter $(BACKTEST_TRUE_VALUES),$(strip $(NO_CACHE))),)
BACKTEST_OPTIONAL_ARGS += --no-cache
endif
ifneq ($(filter $(BACKTEST_TRUE_VALUES),$(strip $(CACHE_WORKING_TREE))),)
BACKTEST_OPTIONAL_ARGS += --cache-working-tree
endif

ifneq ($(strip $(LEAGUES)),)
backtest-expected-points:
	@if [[ -n "$(strip $(FRAME)$(BASELINE_FRAME)$(CANDIDATE_FRAME)$(OUTPUT_DIR))" ]]; then \
		echo "FRAME, BASELINE_FRAME, CANDIDATE_FRAME, and OUTPUT_DIR require a single LEAGUE" >&2; \
		exit 2; \
	fi
	set -a && { [[ ! -f .env ]] || source .env; } && set +a && \
	.venv/bin/python scripts/backtest_expected_points_many.py \
		--leagues "$(LEAGUES)" \
		--profile $(PROFILE) \
		--baseline "$(BASELINE)" \
		--candidate "$(CANDIDATE)" \
		$(BACKTEST_OPTIONAL_ARGS)
else
backtest-expected-points:
	set -a && { [[ ! -f .env ]] || source .env; } && set +a && \
	.venv/bin/python scripts/backtest_expected_points.py compare \
		--league $(LEAGUE) \
		--profile $(PROFILE) \
		--baseline "$(BASELINE)" \
		--candidate "$(CANDIDATE)" \
		$(BACKTEST_FRAME_ARGS) \
		$(BACKTEST_OPTIONAL_ARGS)
endif

replay-expected-points-locks:
	set -a && { [[ ! -f .env ]] || source .env; } && set +a && \
	.venv/bin/python scripts/backtest_expected_points.py lock-replay \
		--league $(LEAGUE) \
		--source "$(LOCK_SOURCE)" \
		$(if $(strip $(LOCK_EXPECTED_VERSION)),--expected-source-version "$(LOCK_EXPECTED_VERSION)",) \
		--minimum-probability $(LOCK_MIN_PROBABILITY) --max-locks-per-week $(LOCK_MAX_PER_WEEK) \
		$(if $(filter $(BACKTEST_TRUE_VALUES),$(strip $(LOCK_SCREEN_CADENCE))),--screen-cadence,) \
		$(if $(strip $(LOCK_EXTRA_MIN_PROBABILITY)),--extra-lock-min-probability $(LOCK_EXTRA_MIN_PROBABILITY),) \
		--bootstrap-samples $(BOOTSTRAP_SAMPLES) \
		$(if $(strip $(OUTPUT_DIR)),--output-dir "$(OUTPUT_DIR)",)

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

prepare-model-release:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py --prepare

sam-deploy:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py

sam-register-releases:
	set -a && source .env && set +a && \
	.venv/bin/python scripts/deploy_models.py --register-only
