# Use the Lambda Python runtime that matches the currently working dependency set.
FROM public.ecr.aws/lambda/python:3.11

# Set the working directory inside the container
WORKDIR /var/task

# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN yum install -y libgomp && \
    yum clean all && \
    pip install setuptools wheel && \
    pip install -r requirements.txt

# LightGBM's OpenMP runtime must reserve its TLS block before Python loads other native modules on arm64.
ENV LD_PRELOAD=/lib64/libgomp.so.1

# Copy the rest of your application code
COPY . .

# Lambda's entry point, using Mangum to adapt FastAPI
CMD ["main.handler"]
