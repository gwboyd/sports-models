# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.12

# Set the working directory inside the container
WORKDIR /var/task

# Install gcc and libomp using dnf, libomp
RUN dnf -y update && \
    dnf -y install gcc gcc-c++ && \ 
    dnf clean all
# Copy the requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the rest of your application code
COPY . .

# Lambda's entry point, using Mangum to adapt FastAPI
CMD ["main.handler"]

