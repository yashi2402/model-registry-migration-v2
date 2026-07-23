"""
Full Model Registry Migration Demo.
Runs the complete pipeline end-to-end:

  1. Connect to Databricks (source) and export models
  2. Migrate to Domino MLflow with metadata preserved
  3. Set up versioning (all versions visible in Models tab)
  4. Run promotion workflow (dev -> staging -> prod with approval)
  5. Show model lineage and experiment tracking
  6. Configure model discoverability (catalog, cards, search)

Source: Databricks (GPS/Davnic MLflow)
Target: Domino MLflow (Models tab, Experiments tab, Data tab)
"""
import os
import sys
from datetime import datetime

from config import DATABRICKS_HOST, DOMINO_MLFLOW_URI, REGISTRY_PATH, EXPERIMENTS_DIR


def run_full_demo():
    """Execute complete migration demo."""
    print("\n" + "#" * 60)
    print("#  MODEL REGISTRY MIGRATION DEMO")
    print("#  Source: Databricks (GPS/Davnic MLflow)")
    print("#  Target: Domino MLflow")
    print("#" * 60)
    print(f"\n  Started: {datetime.now().isoformat()}")
    print(f"  Source: {DATABRICKS_HOST}")
    print(f"  Target: {DOMINO_MLFLOW_URI}")
    print(f"  Registry: {REGISTRY_PATH}")
    print(f"  Experiments: {EXPERIMENTS_DIR}")

    # Step 1 & 2 & 3: Export from Databricks, preserve metadata, set up versioning
    print("\n\n" + "=" * 60)
    print("STEP 1-3: Export from Databricks & Register in Domino")
    print("         (Covers: export, metadata, versioning)")
    print("=" * 60)
    from migrate_models import run_migration
    migration_log = run_migration()

    if not migration_log:
        print("\nERROR: Migration failed. Check Databricks credentials.")
        print("Set environment variables:")
        print("  DATABRICKS_HOST = your workspace URL")
        print("  DATABRICKS_TOKEN = your access token")
        sys.exit(1)

    # Step 4: Model promotion workflow
    print("\n\n" + "=" * 60)
    print("STEP 4: Model Promotion Workflow")
    print("        (dev -> staging -> production with approval)")
    print("=" * 60)
    from model_promotion import run_promotion_workflow
    run_promotion_workflow()

    # Step 5: Model lineage and experiment tracking
    print("\n\n" + "=" * 60)
    print("STEP 5: Model Lineage & Experiment Tracking")
    print("        (Visible in Domino Experiments tab)")
    print("=" * 60)
    from experiment_tracking import show_migrated_lineage
    show_migrated_lineage()

    # Step 6: Model discoverability
    print("\n\n" + "=" * 60)
    print("STEP 6: Model Discoverability & Catalog")
    print("        (Cross-team search and model cards)")
    print("=" * 60)
    from model_discovery import generate_catalog
    generate_catalog()

    # Final summary
    print("\n\n" + "#" * 60)
    print("#  MODEL REGISTRY MIGRATION COMPLETE")
    print("#" * 60)
    print(f"""
Summary:
  [OK] Connected to Databricks and exported {migration_log['total_models']} models
  [OK] Preserved all metadata (params, metrics, tags, artifacts)
  [OK] Registered {migration_log['total_versions']} versions in Domino (Models tab)
  [OK] Demonstrated promotion workflow with approval gate
  [OK] Migrated lineage and experiment tracking (Experiments tab)
  [OK] Generated model catalog and cards (Data tab)

Where to see results in Domino:
  * Models tab       -> Migrated models with versions & stages
  * Experiments tab  -> Migration runs with params & metrics
  * Data tab         -> Catalog, model cards, promotion records

Migration Details:
  * Source:    Databricks ({DATABRICKS_HOST})
  * Target:    Domino MLflow
  * Models:    {migration_log['total_models']}
  * Versions:  {migration_log['total_versions']}
  * Time:      {datetime.now().isoformat()}

Requirement Coverage:
  1. Export models from legacy registry (Databricks)     [OK]
  2. Preserve model metadata                             [OK]
  3. Set up model versioning in Domino                   [OK]
  4. Implement model promotion workflows                 [OK]
  5. Migrate model lineage and experiment tracking       [OK]
  6. Configure model discoverability across teams        [OK]
""")


if __name__ == '__main__':
    run_full_demo()
