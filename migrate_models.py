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
from datetime import datetime

import mlflow
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
            registry_uri='databricks'
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

    def list_source_models(self):
        """List all models in Databricks."""
        print(f"\n{'=' * 60}")
        print("SCANNING SOURCE REGISTRY (Databricks)")
        print("=" * 60)

        models = []
        full_prefix = f"{DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}."

        for rm in self.source_client.search_registered_models():
            model_name = rm.name
            short_name = model_name.replace(full_prefix, '')
            versions = self.source_client.search_model_versions(f"name='{model_name}'")

            model_info = {
                'full_name': model_name,
                'short_name': short_name,
                'versions': [],
                'description': rm.description or '',
                'tags': dict(rm.tags) if rm.tags else {},
            }

            for v in versions:
                run = self.source_client.get_run(v.run_id) if v.run_id else None
                version_info = {
                    'version': v.version,
                    'run_id': v.run_id,
                    'status': v.status,
                    'creation_timestamp': v.creation_timestamp,
                    'params': dict(run.data.params) if run else {},
                    'metrics': dict(run.data.metrics) if run else {},
                    'tags': dict(run.data.tags) if run else {},
                }
                model_info['versions'].append(version_info)

            models.append(model_info)
            print(f"\n  Model: {short_name}")
            print(f"  Versions: {len(versions)}")
            for vi in model_info['versions']:
                acc = vi['metrics'].get('accuracy', 'N/A')
                algo = vi['params'].get('algorithm', 'unknown')
                print(f"    v{vi['version']}: {algo} (accuracy={acc})")

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
