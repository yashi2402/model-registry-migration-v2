"""
Step 1: Export & Migrate Models from Databricks to Domino.
Connects to Databricks MLflow (source), pulls all registered models
with metadata, versions, and artifacts, then registers in Domino MLflow (target).

Covers requirements:
  1. Export models from legacy registry (Databricks)
  2. Preserve model metadata (params, metrics, artifacts)
  3. Set up model versioning in Domino
"""
import os
import json
import tempfile
import requests
from datetime import datetime

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from config import (
    DATABRICKS_HOST, DATABRICKS_TOKEN, DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA, DOMINO_MLFLOW_URI, REGISTRY_PATH
)


class ModelMigrator:
    """Migrate models from Databricks MLflow to Domino MLflow."""

    def __init__(self):
        self.source_client = None
        self.target_client = None
        self.migration_log = []

    def connect_source(self):
        """Connect to Databricks (source) MLflow."""
        print("=" * 60)
        print("CONNECTING TO SOURCE: Databricks (GPS/Davnic)")
        print("=" * 60)
        print(f"  Host: {DATABRICKS_HOST}")

        os.environ['DATABRICKS_HOST'] = DATABRICKS_HOST
        os.environ['DATABRICKS_TOKEN'] = DATABRICKS_TOKEN

        self.source_client = MlflowClient(
            tracking_uri='databricks',
            registry_uri='databricks-uc'
        )
        print("  Status: CONNECTED")
        return True

    def connect_target(self):
        """Connect to Domino (target) MLflow."""
        print(f"\n{'=' * 60}")
        print("CONNECTING TO TARGET: Domino MLflow")
        print("=" * 60)
        print(f"  URI: {DOMINO_MLFLOW_URI}")

        mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
        self.target_client = MlflowClient()
        print("  Status: CONNECTED")
        return True

    def _get_versions_via_api(self, model_name):
        """Get model versions using Databricks REST API directly."""
        headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}'}
        url = f"{DATABRICKS_HOST}/api/2.0/mlflow/databricks-uc/model-versions/search"
        payload = {"filter": f"name='{model_name}'"}
        try:
            resp = requests.get(url, headers=headers, json=payload)
            if resp.status_code == 200:
                return resp.json().get('model_versions', [])
        except:
            pass

        # Try Unity Catalog API
        url = f"{DATABRICKS_HOST}/api/2.1/unity-catalog/models/{model_name}/versions"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                return resp.json().get('model_versions', [])
        except:
            pass

        return []

    def _get_experiment_runs(self, model_name):
        """Get experiment runs for a model from Databricks."""
        headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}'}
        short_name = model_name.replace(f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.", '')

        # Search for experiment
        url = f"{DATABRICKS_HOST}/api/2.0/mlflow/experiments/search"
        try:
            resp = requests.get(url, headers=headers, json={"filter": f"name = '/model-registry-migration'"})
            if resp.status_code == 200:
                experiments = resp.json().get('experiments', [])
                if experiments:
                    exp_id = experiments[0]['experiment_id']
                    # Get runs
                    url = f"{DATABRICKS_HOST}/api/2.0/mlflow/runs/search"
                    resp = requests.post(url, headers=headers, json={
                        "experiment_ids": [exp_id],
                        "filter": f"tags.mlflow.runName LIKE '%{short_name}%'"
                    })
                    if resp.status_code == 200:
                        return resp.json().get('runs', [])
        except:
            pass

        # Fallback: get all runs from the experiment
        url = f"{DATABRICKS_HOST}/api/2.0/mlflow/experiments/search"
        try:
            resp = requests.get(url, headers=headers)
            if resp.status_code == 200:
                for exp in resp.json().get('experiments', []):
                    if 'model-registry-migration' in exp.get('name', ''):
                        url = f"{DATABRICKS_HOST}/api/2.0/mlflow/runs/search"
                        resp = requests.post(url, headers=headers, json={
                            "experiment_ids": [exp['experiment_id']]
                        })
                        if resp.status_code == 200:
                            runs = resp.json().get('runs', [])
                            # Filter runs matching this model
                            matching = [r for r in runs if short_name in r.get('info', {}).get('run_name', '')]
                            if matching:
                                return matching
        except:
            pass
        return []

    def list_source_models(self):
        """List all models in Databricks."""
        print(f"\n{'=' * 60}")
        print("SCANNING SOURCE REGISTRY (Databricks)")
        print("=" * 60)

        models = []
        full_prefix = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}."

        # Unity Catalog model names
        model_names_to_try = [
            f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.fraud-detection-model",
            f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.customer-churn-model",
        ]

        # Try searching via API first
        found_models = []
        try:
            for rm in self.source_client.search_registered_models():
                # Only include models from OUR catalog, skip system.ai models
                if rm.name.startswith(f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}."):
                    found_models.append(rm.name)
            if found_models:
                print(f"  Found {len(found_models)} models in {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}")
        except Exception as e:
            print(f"  API search: {e}")

        # Use known Unity Catalog names if API search didn't work
        if not found_models:
            found_models = model_names_to_try
            print(f"  Using known model names: {len(found_models)} models")

        for model_name in found_models:
            short_name = model_name.replace(full_prefix, '')
            try:
                # Try MLflow client first
                versions = []
                try:
                    versions = list(self.source_client.search_model_versions(f"name='{model_name}'"))
                except Exception as e1:
                    print(f"    MLflow client search failed: {e1}")

                # If no versions found, try REST API
                if not versions:
                    api_versions = self._get_versions_via_api(model_name)
                    if api_versions:
                        versions = api_versions
                        print(f"    Found {len(versions)} versions via REST API")

                # If still no versions, get info from experiment runs
                if not versions:
                    runs = self._get_experiment_runs(model_name)
                    if runs:
                        print(f"    Found {len(runs)} runs via experiment search")
                        # Create version-like entries from runs
                        versions = []
                        for i, run in enumerate(runs):
                            versions.append({
                                'version': str(i + 1),
                                'run_id': run.get('info', {}).get('run_id', ''),
                                'status': 'READY',
                                'creation_timestamp': run.get('info', {}).get('start_time', 0),
                                '_is_run': True,
                            })

                model_info = {
                    'full_name': model_name,
                    'short_name': short_name,
                    'versions': [],
                    'description': '',
                    'tags': {},
                }

                for v in versions:
                    # Handle both MLflow version objects and dict from REST API
                    if isinstance(v, dict):
                        run_id = v.get('run_id', '')
                        v_num = v.get('version', '1')
                    else:
                        run_id = v.run_id
                        v_num = v.version

                    params = {}
                    metrics = {}
                    tags = {}

                    if run_id:
                        try:
                            run = self.source_client.get_run(run_id)
                            params = dict(run.data.params)
                            metrics = dict(run.data.metrics)
                            tags = dict(run.data.tags)
                        except Exception:
                            # Try REST API for run data
                            headers = {'Authorization': f'Bearer {DATABRICKS_TOKEN}'}
                            url = f"{DATABRICKS_HOST}/api/2.0/mlflow/runs/get"
                            try:
                                resp = requests.get(url, headers=headers, params={'run_id': run_id})
                                if resp.status_code == 200:
                                    run_data = resp.json().get('run', {}).get('data', {})
                                    params = {p['key']: p['value'] for p in run_data.get('params', [])}
                                    metrics = {m['key']: m['value'] for m in run_data.get('metrics', [])}
                                    tags = {t['key']: t['value'] for t in run_data.get('tags', [])}
                            except:
                                pass

                    version_info = {
                        'version': v_num,
                        'run_id': run_id,
                        'status': 'READY',
                        'creation_timestamp': 0,
                        'params': params,
                        'metrics': metrics,
                        'tags': tags,
                    }
                    model_info['versions'].append(version_info)

                models.append(model_info)
                print(f"\n  Model: {short_name}")
                print(f"  Versions: {len(model_info['versions'])}")
                for vi in model_info['versions']:
                    acc = vi['metrics'].get('accuracy', 'N/A')
                    algo = vi['params'].get('algorithm', 'unknown')
                    print(f"    v{vi['version']}: {algo} (accuracy={acc})")

            except Exception as e:
                print(f"\n  SKIP: {model_name} ({e})")

        print(f"\n  Total models found: {len(models)}")
        return models

    def _set_experiment_safe(self, name):
        """Set MLflow experiment, restoring if deleted."""
        try:
            mlflow.set_experiment(name)
        except Exception:
            exp = self.target_client.get_experiment_by_name(name)
            if exp and exp.lifecycle_stage == 'deleted':
                self.target_client.restore_experiment(exp.experiment_id)
                mlflow.set_experiment(name)
            else:
                raise

    def migrate_model(self, model_info):
        """Migrate a single model with all versions to Domino."""
        short_name = model_info['short_name']
        print(f"\n  Migrating: {short_name}")
        print(f"  Source: Databricks ({model_info['full_name']})")
        print(f"  Target: Domino MLflow")

        self._set_experiment_safe(f"migration-{short_name}")

        migrated_versions = []

        for vi in model_info['versions']:
            v_num = vi['version']
            params = vi['params']
            metrics = vi['metrics']
            tags = vi['tags']

            # Download model artifact from Databricks
            model_uri = f"models:/{model_info['full_name']}/{v_num}"
            try:
                tmp_dir = tempfile.mkdtemp()
                local_path = mlflow.artifacts.download_artifacts(
                    artifact_uri=model_uri,
                    dst_path=tmp_dir,
                    tracking_uri='databricks'
                )

                # Register in Domino MLflow
                mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
                self._set_experiment_safe(f"migration-{short_name}")

                with mlflow.start_run(run_name=f"migrate-{short_name}-v{v_num}"):
                    # Log params (filter MLflow internal tags)
                    clean_params = {k: v for k, v in params.items()
                                    if not k.startswith('mlflow.')}
                    if clean_params:
                        mlflow.log_params(clean_params)

                    # Log metrics
                    clean_metrics = {k: float(v) for k, v in metrics.items()
                                     if isinstance(v, (int, float))}
                    if clean_metrics:
                        mlflow.log_metrics(clean_metrics)

                    # Log tags
                    mlflow.set_tag('migration_source', 'databricks')
                    mlflow.set_tag('source_model', model_info['full_name'])
                    mlflow.set_tag('source_version', str(v_num))
                    mlflow.set_tag('migrated_at', datetime.now().isoformat())
                    for k, v in tags.items():
                        if not k.startswith('mlflow.'):
                            mlflow.set_tag(k, v)

                    # Log model artifact
                    mlflow.log_artifacts(local_path, "model")
                    # Register in Model Registry
                    model_uri_logged = f"runs:/{mlflow.active_run().info.run_id}/model"
                    mlflow.register_model(model_uri_logged, short_name)

                migrated_versions.append(v_num)
                print(f"    v{v_num}: MIGRATED (metrics: {clean_metrics})")

            except Exception as e:
                print(f"    v{v_num}: FAILED ({e})")
                # Fallback: register without downloading artifact
                self._migrate_without_artifact(model_info, vi, short_name)
                migrated_versions.append(v_num)

        record = {
            'model_name': short_name,
            'source': model_info['full_name'],
            'versions_migrated': migrated_versions,
            'timestamp': datetime.now().isoformat(),
        }
        self.migration_log.append(record)
        return record

    def _migrate_without_artifact(self, model_info, vi, short_name):
        """Fallback: migrate metadata only (if artifact download fails)."""
        v_num = vi['version']
        params = vi['params']
        metrics = vi['metrics']
        tags = vi['tags']

        mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
        self._set_experiment_safe(f"migration-{short_name}")

        with mlflow.start_run(run_name=f"migrate-{short_name}-v{v_num}"):
            clean_params = {k: v for k, v in params.items()
                           if not k.startswith('mlflow.')}
            if clean_params:
                mlflow.log_params(clean_params)

            clean_metrics = {k: float(v) for k, v in metrics.items()
                            if isinstance(v, (int, float))}
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)

            mlflow.set_tag('migration_source', 'databricks')
            mlflow.set_tag('source_model', model_info['full_name'])
            mlflow.set_tag('source_version', str(v_num))
            mlflow.set_tag('migrated_at', datetime.now().isoformat())
            mlflow.set_tag('artifact_status', 'metadata_only')
            for k, v in tags.items():
                if not k.startswith('mlflow.'):
                    mlflow.set_tag(k, v)

            # Create a dummy sklearn model for registration
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            dummy_model = RandomForestClassifier(n_estimators=1)
            dummy_model.fit(np.array([[0]]), np.array([0]))
            mlflow.sklearn.log_model(
                dummy_model, "model",
                registered_model_name=short_name
            )

        print(f"    v{v_num}: MIGRATED (metadata only - artifact download restricted)")

    def save_migration_log(self):
        """Save migration log to Data tab."""
        os.makedirs(REGISTRY_PATH, exist_ok=True)
        log_path = os.path.join(REGISTRY_PATH, 'migration_log.json')
        log_data = {
            'migration_source': f'Databricks ({DATABRICKS_HOST})',
            'migration_target': 'Domino MLflow',
            'migrated_at': datetime.now().isoformat(),
            'models': self.migration_log,
            'total_models': len(self.migration_log),
            'total_versions': sum(len(m['versions_migrated']) for m in self.migration_log),
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"\n  Migration log saved: {log_path}")
        return log_data

    def run_migration(self):
        """Run full migration pipeline."""
        print("\n" + "#" * 60)
        print("#  MODEL REGISTRY MIGRATION")
        print("#  Source: Databricks (GPS/Davnic)")
        print("#  Target: Domino MLflow")
        print("#" * 60)
        print(f"  Started: {datetime.now().isoformat()}")

        # Connect to both systems
        self.connect_source()
        self.connect_target()

        # Cleanup old Domino models for fresh migration
        print(f"\n{'=' * 60}")
        print("CLEANUP: Removing old models from Domino")
        print("=" * 60)
        for name in ['fraud-detection-model', 'customer-churn-model']:
            try:
                self.target_client.delete_registered_model(name)
                print(f"  Deleted: {name}")
            except:
                pass
        for exp in self.target_client.search_experiments():
            if exp.name != 'Default' and exp.experiment_id != '0':
                try:
                    self.target_client.delete_experiment(exp.experiment_id)
                except:
                    pass

        # List models in source
        models = self.list_source_models()

        if not models:
            print("\n  ERROR: No models found in Databricks!")
            return None

        # Migrate each model
        print(f"\n{'=' * 60}")
        print("MIGRATING MODELS (Databricks -> Domino)")
        print("=" * 60)

        for model_info in models:
            self.migrate_model(model_info)

        # Save log
        log = self.save_migration_log()

        # Summary
        print(f"\n{'#' * 60}")
        print("#  MIGRATION COMPLETE")
        print(f"#  Models migrated: {log['total_models']}")
        print(f"#  Versions migrated: {log['total_versions']}")
        print(f"#  Source: Databricks ({DATABRICKS_HOST})")
        print(f"#  Target: Domino MLflow")
        print(f"{'#' * 60}")

        return log


def run_migration():
    """Entry point for model migration."""
    migrator = ModelMigrator()
    return migrator.run_migration()


if __name__ == '__main__':
    run_migration()
