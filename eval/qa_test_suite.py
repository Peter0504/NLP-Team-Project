import json
import psycopg2
from sklearn.metrics import accuracy_score, f1_score

# Dummy ground truth data for 10 transcripts
# In reality, you would manually label 10 transcripts to test against
GROUND_TRUTH_FILE = "eval/ground_truth_10_calls.json"
BASELINE_FILE = "eval/baseline_metrics.json"

def load_ground_truth():
    with open(GROUND_TRUTH_FILE, 'r') as f:
        return json.load(f)

def run_qa_suite(db_config):
    truth_data = load_ground_truth()
    y_true_sentiment = []
    y_pred_sentiment = []
    topic_matches = 0
    total_topics = 0

    conn = psycopg2.connect(**db_config)
    
    with conn.cursor() as cur:
        for call_id, expected in truth_data.items():
            # Fetch the predicted NLP data from your Postgres table
            cur.execute("""
                SELECT sentiment, topics 
                FROM transcript_chunks 
                WHERE call_id = %s
            """, (call_id,))
            
            results = cur.fetchall()
            
            # Aggregate predictions (simplified logic)
            # Assuming you take the majority sentiment of the chunks
            pred_sentiments = [r[0] for r in results if r[0] is not None]
            if not pred_sentiments:
                continue
                
            majority_sentiment = max(set(pred_sentiments), key=pred_sentiments.count)
            
            y_true_sentiment.append(expected['overall_sentiment'])
            y_pred_sentiment.append(majority_sentiment)
            
            # Check topic precision (did the model find the expected topic?)
            pred_topics = [r[1] for r in results if r[1] is not None]
            if expected['expected_topic'] in pred_topics:
                topic_matches += 1
            total_topics += 1

    # Calculate Metrics
    accuracy = accuracy_score(y_true_sentiment, y_pred_sentiment)
    f1 = f1_score(y_true_sentiment, y_pred_sentiment, average='weighted')
    topic_precision = topic_matches / total_topics if total_topics > 0 else 0

    metrics = {
        "sentiment_accuracy": accuracy,
        "sentiment_f1": f1,
        "topic_precision": topic_precision
    }

    # Record Baseline
    with open(BASELINE_FILE, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"QA Suite Complete. Baseline recorded: {metrics}")

if __name__ == "__main__":
    DB_CONFIG = {"host": "localhost", "dbname": "stocksense", "user": "postgres"}
    run_qa_suite(DB_CONFIG)