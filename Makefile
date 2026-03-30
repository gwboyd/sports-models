SHELL := /bin/zsh

.PHONY: help backend frontend sam-build sam-invoke-health sam-api

help:
	@echo "Available targets:"
	@echo "  make backend            Run FastAPI locally on 127.0.0.1:3000"
	@echo "  make frontend           Run Remix locally on 127.0.0.1:5173"
	@echo "  make sam-build          Build the Lambda image/artifacts with SAM"
	@echo "  make sam-invoke-health  Invoke the SAM Lambda with the sample health event"
	@echo "  make sam-api            Run the SAM local API on 127.0.0.1:3001"

backend:
	source .venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 3000 --reload

frontend:
	cd frontend && npm run dev

sam-build:
	sam build

sam-invoke-health:
	set -a && source .env && set +a && \
	sam local invoke FastAPILambdaFunction \
		-e events/get-health-event.json \
		--parameter-overrides \
			Localhost=True \
			EnvironmentName=LOCAL \
			AdminApiKey="$$ADMIN_API_KEY" \
			FrontEndApiKey="$$FRONT_END_API_KEY" \
			ReadApiKey="$$READ_API_KEY" \
			NbaApiKey="$$NBA_API_KEY" \
			AwsApiKey="$$AWS_API_KEY" \
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
			SupabaseDbUrl="$$SUPABASE_DB_URL" \
			SupabaseSchema="$$SUPABASE_SCHEMA"
