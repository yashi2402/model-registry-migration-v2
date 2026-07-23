"""
Step 2: Model Promotion Workflow.
Implements dev -> staging -> production pipeline with:
  - Validation gates (metric thresholds)
  - Interactive approval for production
  - Email notification after decision
  - MLflow stage transitions (visible in Domino Models tab)

Covers requirement 4: Implement model promotion workflows
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

from config import DOMINO_MLFLOW_URI, REGISTRY_PATH, PROMOTION_STAGES, APPROVER_EMAIL


def send_notification_email(subject, body, to_email=APPROVER_EMAIL):
    """Send email notification about promotion decision."""
    try:
        msg = MIMEMultipart()
        msg['From'] = 'model-registry@domino.tech'
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('localhost', 25, timeout=5) as server:
            server.sendmail(msg['From'], to_email, msg.as_string())
        print(f"  [EMAIL] Notification sent to {to_email}")
    except Exception as e:
        print(f"  [EMAIL] Notification skipped (SMTP not configured: {type(e).__name__})")
        print(f"  [EMAIL] Would have sent to: {to_email}")
        print(f"  [EMAIL] Subject: {subject}")


class ModelPromotionManager:
    """Manages model promotion with validation gates and approval."""

    def __init__(self):
        mlflow.set_tracking_uri(DOMINO_MLFLOW_URI)
        self.client = MlflowClient()
        self.promotion_rules = {
            'staging': {
                'min_accuracy': 0.70,
                'requires_approval': False,
            },
            'production': {
                'min_accuracy': 0.80,
                'requires_approval': True,
                'approvers': [APPROVER_EMAIL],
            },
        }

    def get_model_metrics(self, model_name, version):
        """Get metrics for a model version from MLflow."""
        versions = self.client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            if v.version == str(version):
                run = self.client.get_run(v.run_id)
                return dict(run.data.metrics)
        return {}

    def validate_promotion(self, model_name, version, target_stage):
        """Check if model meets promotion criteria."""
        metrics = self.get_model_metrics(model_name, version)
        rules = self.promotion_rules.get(target_stage, {})
        checks = []
        passed = True

        for metric_key in ['min_accuracy', 'min_f1_score']:
            if metric_key in rules:
                metric_name = metric_key.replace('min_', '')
                actual = metrics.get(metric_name, 0)
                required = rules[metric_key]
                check_passed = actual >= required
                checks.append({
                    'metric': metric_name,
                    'required': required,
                    'actual': round(actual, 4),
                    'passed': check_passed,
                })
                if not check_passed:
                    passed = False

        return {
            'valid': passed,
            'checks': checks,
            'requires_approval': rules.get('requires_approval', False),
            'approvers': rules.get('approvers', []),
            'metrics': metrics,
        }

    def promote(self, model_name, version, target_stage, reason=''):
        """Promote model with validation and approval."""
        print(f"\n--- Promoting: {model_name} v{version} -> {target_stage} ---")

        validation = self.validate_promotion(model_name, version, target_stage)

        # Show validation results
        for check in validation['checks']:
            status = "PASS" if check['passed'] else "FAIL"
            print(f"  [{status}] {check['metric']}: {check['actual']} (min: {check['required']})")

        if not validation['valid']:
            print(f"  BLOCKED: Failed validation checks")
            return {'promoted': False, 'validation': validation}

        # Interactive approval for production
        if validation['requires_approval']:
            approvers = validation['approvers']
            print(f"\n  APPROVAL REQUIRED from: {', '.join(approvers)}")
            print(f"  Model: {model_name} v{version}")
            print(f"  Target: {target_stage}")
            print(f"  Reason: {reason}")

            approval = input(f"\n  [{approvers[0]}] Do you approve this promotion? (yes/no): ").strip().lower()

            if approval not in ('yes', 'y'):
                print(f"  REJECTED by {approvers[0]}")
                send_notification_email(
                    subject=f"[REJECTED] Model Promotion: {model_name} v{version} -> {target_stage}",
                    body=f"Model promotion REJECTED.\n\nModel: {model_name}\nVersion: {version}\nTarget: {target_stage}\nReason: {reason}\nRejected by: {approvers[0]}\nTime: {datetime.now().isoformat()}"
                )
                self._save_record(model_name, version, target_stage, False, reason, approvers[0])
                return {'promoted': False, 'rejected_by': approvers[0]}

            print(f"  APPROVED by {approvers[0]}")
            send_notification_email(
                subject=f"[APPROVED] Model Promotion: {model_name} v{version} -> {target_stage}",
                body=f"Model promotion APPROVED.\n\nModel: {model_name}\nVersion: {version}\nTarget: {target_stage}\nReason: {reason}\nApproved by: {approvers[0]}\nTime: {datetime.now().isoformat()}"
            )

        # Transition stage in MLflow
        stage_map = {'staging': 'Staging', 'production': 'Production', 'archived': 'Archived'}
        mlflow_stage = stage_map.get(target_stage, 'None')

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=str(version),
                stage=mlflow_stage
            )
            print(f"  Stage updated: {mlflow_stage} (visible in Domino Models tab)")
        except Exception as e:
            print(f"  Stage transition note: {e}")

        self._save_record(model_name, version, target_stage, True, reason, APPROVER_EMAIL)
        return {'promoted': True, 'validation': validation}

    def _save_record(self, model_name, version, target_stage, promoted, reason, approver):
        """Save promotion record to Data tab."""
        records_dir = os.path.join(REGISTRY_PATH, 'promotion_records')
        os.makedirs(records_dir, exist_ok=True)

        record = {
            'model_name': model_name,
            'version': version,
            'target_stage': target_stage,
            'promoted': promoted,
            'reason': reason,
            'decided_by': approver,
            'timestamp': datetime.now().isoformat(),
        }

        status = 'APPROVED' if promoted else 'REJECTED'
        filename = f"{model_name}_v{version}_{target_stage}_{status}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(records_dir, filename), 'w') as f:
            json.dump(record, f, indent=2)


def run_promotion_workflow():
    """Demonstrate promotion workflow on migrated models."""
    print("\n" + "=" * 60)
    print("MODEL PROMOTION WORKFLOW")
    print("(dev -> staging -> production)")
    print("=" * 60)

    manager = ModelPromotionManager()

    # List available models
    print("\nRegistered models in Domino:")
    for rm in manager.client.search_registered_models():
        versions = manager.client.search_model_versions(f"name='{rm.name}'")
        print(f"  {rm.name} ({len(versions)} versions)")

    # Promote fraud-detection v2 to staging (auto-approved)
    manager.promote(
        'fraud-detection-model', 2, 'staging',
        reason='GradientBoosting v2 outperforms v1 RandomForest'
    )

    # Promote fraud-detection v2 to production (requires approval)
    manager.promote(
        'fraud-detection-model', 2, 'production',
        reason='Passed staging validation, ready for production'
    )

    # Promote customer-churn v2 to staging (auto-approved)
    manager.promote(
        'customer-churn-model', 2, 'staging',
        reason='GradientBoosting improves over LogisticRegression baseline'
    )

    print(f"\n{'=' * 60}")
    return manager


if __name__ == '__main__':
    run_promotion_workflow()
