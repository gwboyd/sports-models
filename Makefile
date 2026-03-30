SHELL := /bin/zsh

.PHONY: help backend frontend frontend-local frontend-sam sam-build sam-invoke-health sam-api sam-deploy

help:
	@echo "Available targets:"
	@echo "  make backend            Run FastAPI locally on 127.0.0.1:3000"
	@echo "  make frontend           Run Remix using frontend/.env as-is"
	@echo "  make frontend-local     Run Remix against direct local backend (127.0.0.1:3000)"
	@echo "  make frontend-sam       Run Remix against local SAM API (127.0.0.1:3001)"
	@echo "  make sam-build          Build the Lambda image/artifacts with SAM"
	@echo "  make sam-invoke-health  Invoke the API Lambda with the sample health event"
	@echo "  make sam-api            Run the SAM local API on 127.0.0.1:3001"
	@echo "  make sam-deploy         Deploy the sports-models-v2 stack to us-east-1"

backend:
	source .venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 3000 --reload

frontend:
	cd frontend && npm run dev

frontend-local:
	cd frontend && ENDPOINT=http://127.0.0.1:3000 npm run dev

frontend-sam:
	cd frontend && ENDPOINT=http://127.0.0.1:3001 npm run dev

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

sam-deploy:
	set -a && source .env && set +a && \
	sam build && \
	sam deploy \
		--stack-name sports-models-v2 \
		--region us-east-1 \
		--resolve-s3 \
		--resolve-image-repos \
		--capabilities CAPABILITY_IAM \
		--no-confirm-changeset \
		--no-fail-on-empty-changeset \
		--parameter-overrides \
			HttpApiName=sports-models-http-api-v2 \
			ApiFunctionName=sports-models-api-v2 \
			TrainingFunctionName=sports-models-training-v2 \
			Localhost=False \
			EnvironmentName=PROD \
			AdminApiKey="$$ADMIN_API_KEY" \
			FrontEndApiKey="$$FRONT_END_API_KEY" \
			ReadApiKey="$$READ_API_KEY" \
			NbaApiKey="$$NBA_API_KEY" \
			AwsApiKey="$$AWS_API_KEY" \
			SupabaseDbUrl="$$SUPABASE_DB_URL" \
			SupabaseSchema="$$SUPABASE_SCHEMA"
