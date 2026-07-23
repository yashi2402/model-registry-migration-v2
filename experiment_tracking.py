"""
Step 3: Experiment Tracking & Model Lineage.
Shows that experiment history from Databricks is preserved in Domino's
Experiments tab, and demonstrates lineage tracking.

Covers requirement 5: Migrate model lineage and experiment tracking
"""
import os
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from config import DOMINO_MLFLOW_URI, DATABRICKS_HOST, EXPERIMENTS_DIR, REGISTRY_PATH


def show_migrated_lineage():
    """Display lineage from migrated models - shows experiment history in Domino."""
    print("\n" + "=" * 60)
    print("MODEL LINEAGE & EXPERIMENT TRACKING")
    print("(Visible in Domino Experiments tab)")
    print("=" * 60)

    mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
    client = MlflowClient()

    lineage_data = {
        'source_system': f'Databricks ({DATABRICKS_HOST})',
        'target_system': 'Domino MLflow',
        'migrated_at': datetime.now().isoformat(),
        'models': [],
    }

    # Show experiments (migration runs)
    experiments = client.search_experiments()
    migration_exps = [e for e in experiments if 'dbx-migration' in e.name.lower()]

    print(f"\n  Migration experiments found: {len(migration_exps)}")

    for exp in migration_exps:
        print(f"\n  Experiment: {exp.name}")
        runs = client.search_runs(experiment_ids=[exp.experiment_id])

        model_lineage = {
            'experiment': exp.name,
            'total_runs': len(runs),
            'runs': [],
        }

        for run in runs:
            run_info = {
                'run_name': run.info.run_name,
                'run_id': run.info.run_id,
                'status': run.info.status,
                'params': dict(run.data.params),
                'metrics': {k: round(v, 4) for k, v in run.data.metrics.items()},
                'tags': {k: v for k, v in run.data.tags.items() if not k.startswith('mlflow.')},
            }
            model_lineage['runs'].append(run_info)

            acc = run.data.metrics.get('accuracy', 'N/A')
            algo = run.data.params.get('algorithm', run.data.params.get('framework', ''))
            source_ver = run.data.tags.get('source_version', '?')
            print(f"    Run: {run.info.run_name}")
            print(f"      Source: Databricks v{source_ver}")
            print(f"      Algorithm: {algo}")
            print(f"      Accuracy: {acc}")
            print(f"      Migrated: {run.data.tags.get('migrated_at', 'N/A')}")

        lineage_data['models'].append(model_lineage)

    # Save lineage report
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    lineage_path = os.path.join(EXPERIMENTS_DIR, 'lineage_report.json')
    with open(lineage_path, 'w') as f:
        json.dump(lineage_data, f, indent=2)

    print(f"\n  Lineage report saved: {lineage_path}")

    # Show best models per experiment
    print(f"\n  --- Best Model Per Experiment ---")
    for model in lineage_data['models']:
        if model['runs']:
            best = max(model['runs'], key=lambda r: r['metrics'].get('accuracy', 0))
            print(f"  {model['experiment']}: {best['run_name']} (accuracy={best['metrics'].get('accuracy', 'N/A')})")

    print(f"\n{'=' * 60}")
    return lineage_data


if __name__ == '__main__':
    show_migrated_lineage()
