import subprocess
import sys
import time

def run_script(script_path):
    """Runs a python script and halts execution if it fails."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Execute the script using the current Python interpreter
    result = subprocess.run([sys.executable, script_path])
    
    elapsed_time = time.time() - start_time
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {script_path} failed with exit code {result.returncode}.")
        print("Halting pipeline execution.")
        sys.exit(1)
        
    print(f"\n✅ SUCCESS: {script_path} completed in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    # Define the sequential order of the batch jobs
    pipeline_scripts = [
        #"data_prep/step1_ingestion.py",
        #"data_prep/step2_chunking.py",
        #"pipeline/step3_enrichment.py",
        "pipeline/step4_indexing.py",
        "eval/qa_test_suite.py"
    ]
    
    print("Starting the StockSense End-to-End Pipeline...")
    
    for script in pipeline_scripts:
        run_script(script)
        
    print("\n🎉 All batch processing, indexing, and evaluations are complete!")
    print("\n" + "*"*60)
    print("🌐 NEXT STEP: START THE API SERVER")
    print("*"*60)
    print("Run the following command in your terminal to launch FastAPI:")
    print("uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")