import json
import psycopg2
from datetime import datetime, timedelta

BASELINE_FILE = "eval/baseline_metrics.json"
DRIFT_THRESHOLD = 0.05 # 5% drop allowed

def load_baseline():
    with open(BASELINE_FILE, 'r') as f:
        return json.load(f)

def trigger_retraining_alert(metric_name, baseline_val, current_val):
    drop = (baseline_val - current_val) / baseline_val
    print(f"🚨 ALERT: Model drift detected in {metric_name}!")
    print(f"Baseline: {baseline_val:.2f} | Current: {current_val:.2f} | Drop: {drop:.1%}")
    print("Initiating retraining pipeline (or sending Slack/Email alert)...")
    # Add your Webhook or Airflow trigger here

def check_model_drift(db_config):
    baseline = load_baseline()
    
    conn = psycopg2.connect(**db_config)
    
    # Calculate the date 3 months ago
    three_months_ago = datetime.now() - timedelta(days=90)
    
    with conn.cursor() as cur:
        # Fetch the model's confidence scores over the last 3 months
        # A drop in average confidence is a common proxy for drift when ground-truth labels aren't available live
        cur.execute("""
            SELECT AVG((metadata->>'sentiment_score')::float) as avg_confidence
            FROM transcripts
            WHERE call_date >= %s
        """, (three_months_ago.date(),))
        
        result = cur.fetchone()
        avg_recent_confidence = result[0] if result[0] else 0

    conn.close()

    # Compare against a baseline metric (e.g., assuming baseline had a target confidence or accuracy)
    # If using confidence as a proxy for accuracy drift:
    target_confidence = baseline.get('sentiment_accuracy', 0.85) # Example fallback
    
    if avg_recent_confidence < (target_confidence * (1 - DRIFT_THRESHOLD)):
        trigger_retraining_alert("Sentiment Confidence", target_confidence, avg_recent_confidence)
    else:
        print(f"✅ Models are healthy. Current avg confidence ({avg_recent_confidence:.2f}) is within threshold.")

if __name__ == "__main__":
    DB_CONFIG = {"host": "localhost", "dbname": "stocksense", "user": "postgres"}
    print(f"Running Drift Check for 3-month window ending {datetime.now().date()}...")
    check_model_drift(DB_CONFIG)