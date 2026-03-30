# Use the Lambda Python runtime that matches the currently working dependency set.
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory inside the container
WORKDIR /var/task

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install setuptools wheel && \
    pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Lambda's entry point, using Mangum to adapt FastAPI
CMD ["main.handler"]
