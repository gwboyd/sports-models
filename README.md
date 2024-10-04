# sports-models

Welcome to Will's **sports-model** – a set of models that can be used to predict the outcomes of sporting events.

## Getting Started

To ensure a clean, isolated Python environment, it is highly recommended to create a virtual environment before proceeding with project setup.

### Setting up a Virtual Environment

#### Step 1: Create a Virtual Environment

Make sure you are using python 3.12.5

Create a virtual environment folder `.venv` in the root directory of the repository:

```shell
python3 -m venv .venv
```

Alternatively, you can setup the virtual environment using Visual Studio Code (VSCode).

#### Step 2: Activate the Virtual Environment

On Unix or MacOS, using the terminal:

```shell
source .venv/bin/activate
```

### Installing Dependencies

#### Step 3: Install OpenMP for xgboost

```shell
brew install libomp
```

#### Step 4: Update Python Certification


```shell
brew install python-certifi
```

#### Step 4: Install All Other Required Packages
```shell
pip3 install -r requirements.txt
```

### Step 5: Configure Dynamo DB set up

Reach out to Will to give you an AWS IAM role for the tables