"""
Configuration for Model Registry Migration.
Databricks (source) -> Domino (target)
"""
import os

# Source: Databricks (GPS/Davnic MLflow)
DATABRICKS_HOST = os.environ.get('DATABRICKS_HOST', 'https://adb-7405619830961914.14.azuredatabricks.net')
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN', '')
DATABRICKS_CATALOG = 'migvisor_india'
DATABRICKS_SCHEMA = 'default'

# Target: Domino MLflow
DOMINO_MLFLOW_URI = os.environ.get('MLFLOW_TRACKING_URI', '')

# Domino environment
DOMINO_PROJECT_OWNER = os.environ.get('DOMINO_PROJECT_OWNER', '')
DOMINO_PROJECT_NAME = os.environ.get('DOMINO_PROJECT_NAME', '')

# Paths (Domino data mount)
REGISTRY_PATH = '/mnt/data/model-registry/registry'
EXPERIMENTS_DIR = '/mnt/data/model-registry/experiments'

# Model promotion stages
PROMOTION_STAGES = ['development', 'staging', 'production', 'archived']

# Approver for production promotions
APPROVER_EMAIL = 'yashi_rahangdale@epam.com'
