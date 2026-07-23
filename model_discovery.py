"""
Step 4: Model Discoverability & Cross-Team Sharing.
Creates searchable catalog, model cards, and configures
models for cross-team discovery.

Covers requirement 6: Configure model discoverability across teams
"""
import os
import json
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from config import DOMINO_MLFLOW_URI, REGISTRY_PATH, DATABRICKS_HOST


def generate_catalog():
    """Generate searchable model catalog from migrated models."""
    print("\n" + "=" * 60)
    print("MODEL DISCOVERABILITY & CATALOG")
    print("=" * 60)

    mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
    client = MlflowClient()

    catalog = {
        'catalog_name': 'Model Registry Catalog',
        'source': f'Migrated from Databricks ({DATABRICKS_HOST})',
        'generated_at': datetime.now().isoformat(),
        'models': [],
    }

    print("\n[1] Building model catalog from Domino MLflow Registry...")

    for rm in client.search_registered_models():
        versions = client.search_model_versions(f"name='{rm.name}'")
        latest = max(versions, key=lambda v: int(v.version)) if versions else None

        # Get metadata from latest version's run
        metadata = {}
        if latest and latest.run_id:
            run = client.get_run(latest.run_id)
            metadata = {
                'params': dict(run.data.params),
                'metrics': {k: round(v, 4) for k, v in run.data.metrics.items()},
                'tags': {k: v for k, v in run.data.tags.items() if not k.startswith('mlflow.')},
            }

        model_entry = {
            'name': rm.name,
            'description': rm.description or '',
            'latest_version': latest.version if latest else '0',
            'total_versions': len(versions),
            'stage': latest.current_stage if latest else 'None',
            'owner': metadata.get('tags', {}).get('team', 'unknown'),
            'domain': metadata.get('tags', {}).get('domain', 'unknown'),
            'algorithm': metadata.get('params', {}).get('algorithm', 'unknown'),
            'framework': metadata.get('params', {}).get('framework', 'unknown'),
            'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
            'f1_score': metadata.get('metrics', {}).get('f1_score', 0),
            'migration_source': metadata.get('tags', {}).get('migration_source', 'unknown'),
        }
        catalog['models'].append(model_entry)

    # Print catalog table
    print(f"\n[2] Model Catalog ({len(catalog['models'])} models):")
    print(f"    {'Name':<30} {'Stage':<12} {'Algorithm':<20} {'Team':<15} {'Accuracy':<10}")
    print(f"    {'-'*30} {'-'*12} {'-'*20} {'-'*15} {'-'*10}")
    for m in catalog['models']:
        print(f"    {m['name']:<30} {m['stage']:<12} {m['algorithm']:<20} {m['owner']:<15} {m['accuracy']:<10}")

    # Search demo
    print(f"\n[3] Search Demo:")
    print(f"    Search: 'fraud'")
    results = [m for m in catalog['models'] if 'fraud' in m['name'].lower()]
    for r in results:
        print(f"    Found: {r['name']} (v{r['latest_version']}, {r['stage']})")

    print(f"\n    Search: domain='payments'")
    results = [m for m in catalog['models'] if m['domain'] == 'payments']
    for r in results:
        print(f"    Found: {r['name']} (team: {r['owner']})")

    # Generate model cards
    print(f"\n[4] Generating Model Cards...")
    cards_dir = os.path.join(REGISTRY_PATH, 'model_cards')
    os.makedirs(cards_dir, exist_ok=True)

    model_descriptions = {
        'fraud-detection-model': {
            'purpose': 'Detects fraudulent credit card transactions in real-time',
            'input': 'Transaction features (amount, merchant, time, location, card info)',
            'output': 'Binary classification (fraud/not fraud) with probability score',
            'stakeholders': 'Payments team, Risk & Compliance',
            'sla': '< 100ms inference latency for real-time scoring',
        },
        'customer-churn-model': {
            'purpose': 'Predicts customer churn probability for proactive retention',
            'input': 'Customer features (tenure, charges, support tickets, usage)',
            'output': 'Churn probability score (0-1)',
            'stakeholders': 'Analytics team, Customer Success',
            'sla': 'Daily batch scoring for all active customers',
        },
    }

    for model in catalog['models']:
        desc = model_descriptions.get(model['name'], {})
        card = {
            'model_name': model['name'],
            'version': model['latest_version'],
            'description': desc.get('purpose', model['description']),
            'input_data': desc.get('input', 'See model documentation'),
            'output': desc.get('output', 'Classification result'),
            'stakeholders': desc.get('stakeholders', 'TBD'),
            'sla': desc.get('sla', 'TBD'),
            'algorithm': model['algorithm'],
            'framework': model['framework'],
            'metrics': {'accuracy': model['accuracy'], 'f1_score': model['f1_score']},
            'owner_team': model['owner'],
            'domain': model['domain'],
            'migration_source': model['migration_source'],
            'stage': model['stage'],
            'generated_at': datetime.now().isoformat(),
        }

        card_path = os.path.join(cards_dir, f"{model['name']}_card.json")
        with open(card_path, 'w') as f:
            json.dump(card, f, indent=2)
        print(f"    Card: {model['name']} - {desc.get('purpose', 'No description')[:60]}...")

    # Save catalog
    catalog_path = os.path.join(REGISTRY_PATH, 'model_catalog.json')
    with open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    # Group by team
    print(f"\n[5] Models by Team:")
    teams = {}
    for m in catalog['models']:
        teams.setdefault(m['owner'], []).append(m['name'])
    for team, models in teams.items():
        print(f"    {team}: {', '.join(models)}")

    # Group by domain
    print(f"\n[6] Models by Domain:")
    domains = {}
    for m in catalog['models']:
        domains.setdefault(m['domain'], []).append(m['name'])
    for domain, models in domains.items():
        print(f"    {domain}: {', '.join(models)}")

    print(f"\n  Catalog saved: {catalog_path}")
    print(f"  Model cards saved: {cards_dir}/")
    print(f"\n  TIP: Turn on 'Globally Discoverable' in Domino Models tab")
    print(f"       for cross-project visibility")
    print(f"\n{'=' * 60}")

    return catalog


if __name__ == '__main__':
    generate_catalog()
