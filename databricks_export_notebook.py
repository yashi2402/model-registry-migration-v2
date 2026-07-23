# Databricks notebook source
# TITLE: Export Models for Migration to Domino
# IMPORTANT: Run this on a CLUSTER (not serverless)
# This exports actual model artifacts (pickle files) to DBFS
# so Domino can download them via REST API

import mlflow
import json
import os
from mlflow.tracking import MlflowClient

# Configuration
CATALOG = "migvisor_india"
SCHEMA = "default"
LOCAL_EXPORT = "/tmp/model-exports"
DBFS_EXPORT = "dbfs:/model-exports"

client = MlflowClient(registry_uri="databricks-uc")

# Find all models in our catalog
models_to_export = []
for rm in client.search_registered_models():
    if rm.name.startswith(f"{CATALOG}.{SCHEMA}."):
        models_to_export.append(rm.name)

print(f"Found {len(models_to_export)} models to export")
print(f"Models: {[m.split('.')[-1] for m in models_to_export]}")

export_manifest = {
    "source": "databricks",
    "host": spark.conf.get("spark.databricks.workspaceUrl", ""),
    "catalog": CATALOG,
    "schema": SCHEMA,
    "models": []
}

for model_name in models_to_export:
    short_name = model_name.replace(f"{CATALOG}.{SCHEMA}.", "")
    print(f"\n{'='*50}")
    print(f"Exporting: {short_name}")
    print(f"{'='*50}")

    versions = list(client.search_model_versions(f"name='{model_name}'"))
    print(f"  Versions found: {len(versions)}")

    model_export = {
        "full_name": model_name,
        "short_name": short_name,
        "versions": []
    }

    for mv in versions:
        v_num = mv.version
        run_id = mv.run_id
        print(f"\n  Version {v_num} (run_id: {run_id})")

        # Get run data
        run = client.get_run(run_id)
        params = dict(run.data.params)
        metrics = dict(run.data.metrics)
        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}

        # Download artifact to /tmp/
        export_dir = f"{LOCAL_EXPORT}/{short_name}/v{v_num}"
        os.makedirs(export_dir, exist_ok=True)

        print(f"  Downloading artifact...")
        artifact_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name}/{v_num}",
            dst_path=export_dir
        )
        print(f"  Downloaded to: {artifact_path}")

        # List downloaded files
        for root, dirs, files in os.walk(export_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                fsize = os.path.getsize(fpath) / 1024
                print(f"    File: {os.path.relpath(fpath, export_dir)} ({fsize:.1f} KB)")

        # Save metadata
        metadata = {
            "model_name": model_name,
            "short_name": short_name,
            "version": v_num,
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
            "tags": tags,
            "status": mv.status,
            "creation_timestamp": mv.creation_timestamp,
            "description": mv.description or ""
        }
        metadata_path = f"{export_dir}/migration_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata saved")

        # Copy to DBFS
        dbfs_dest = f"{DBFS_EXPORT}/{short_name}/v{v_num}"
        dbutils.fs.cp(f"file:{export_dir}", dbfs_dest, recurse=True)
        print(f"  Copied to DBFS: {dbfs_dest}")

        model_export["versions"].append({
            "version": v_num,
            "dbfs_path": f"/model-exports/{short_name}/v{v_num}",
            "run_id": run_id,
            "params": params,
            "metrics": metrics,
            "tags": tags
        })

    export_manifest["models"].append(model_export)

# Save manifest
manifest_local = f"{LOCAL_EXPORT}/export_manifest.json"
with open(manifest_local, "w") as f:
    json.dump(export_manifest, f, indent=2)
dbutils.fs.cp(f"file:{manifest_local}", f"{DBFS_EXPORT}/export_manifest.json")

print(f"\n{'='*60}")
print(f"EXPORT COMPLETE")
print(f"{'='*60}")
print(f"Models exported: {len(export_manifest['models'])}")
print(f"DBFS location: {DBFS_EXPORT}/")
print(f"Manifest: {DBFS_EXPORT}/export_manifest.json")

# Verify
print(f"\nFiles in DBFS:")
for f in dbutils.fs.ls(DBFS_EXPORT):
    print(f"  {f.path}")
    if f.isDir():
        for sf in dbutils.fs.ls(f.path):
            print(f"    {sf.path}")
            if sf.isDir():
                for ssf in dbutils.fs.ls(sf.path):
                    print(f"      {ssf.name} ({ssf.size} bytes)")
