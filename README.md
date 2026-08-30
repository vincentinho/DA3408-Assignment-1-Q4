# Reproducibility Capstone

## Reproduce the experiment

### 1. Clone the repository

git clone <repository>
cd reproducibility-capstone

### 2. Checkout the exact commit

git checkout <COMMIT>

### 3. Configure DVC

Set the following environment variables:

AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY

### 4. Pull the exact dataset

dvc checkout

### 5. Create the Python environment

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

### 6. Start MLflow

mlflow server --host 127.0.0.1 --port 5000

### 7. Run the experiment

python src/train.py

### Expected result

Accuracy should be approximately:

0.9666666667

Tolerance:

±0.000001
