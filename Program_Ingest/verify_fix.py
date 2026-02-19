import time
import requests
import subprocess
import sys
import os
import signal

def verify():
    # Start API
    print("Starting API...")
    # Using a different port to avoid conflicts
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", "8123"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/revan/docker/APIinterface/Holonet"
    )
    
    # Wait for startup
    # We will poll the health endpoint
    api_up = False
    for i in range(10):
        time.sleep(2)
        try:
            resp = requests.get("http://localhost:8123/health")
            if resp.status_code == 200:
                print("API is up!")
                api_up = True
                break
        except requests.exceptions.ConnectionError:
            print(f"Waiting for API... ({i+1}/10)")
            continue
            
    if not api_up:
        print("API failed to start.")
        stdout, stderr = proc.communicate()
        print("STDOUT:", stdout.decode())
        print("STDERR:", stderr.decode())
        return

    try:
        # Test 1: Health Check
        print("\n--- Test 1: Health Check ---")
        health = requests.get("http://localhost:8123/health").json()
        print(f"Nodes in KB: {health.get('kb_nodes')}")
        if health.get('kb_nodes', 0) > 0:
            print("PASS: KB loaded.")
        else:
            print("WARNING: KB empty (might be expected if no files).")

        # Test 2: Query with max_nodes = 1
        print("\n--- Test 2: Query with max_nodes=1 ---")
        payload = {"query": "What are the core principles?", "max_nodes": 1}
        resp = requests.post("http://localhost:8123/query", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get('retrieved_nodes', [])
            print(f"Retrieved {len(nodes)} nodes.")
            if len(nodes) <= 1:
                print("PASS: max_nodes respected.")
            else:
                print("FAIL: Retrieved more than 1 node.")
        else:
            print(f"FAIL: Query failed ({resp.status_code}) - {resp.text}")

        # Test 3: Query with default (should be all/4)
        print("\n--- Test 3: Query with default max_nodes ---")
        payload = {"query": "What are the core principles?"}
        resp = requests.post("http://localhost:8123/query", json=payload)
        if resp.status_code == 200:
            data = resp.json()
            nodes = data.get('retrieved_nodes', [])
            print(f"Retrieved {len(nodes)} nodes.")
            if len(nodes) > 1:
                 print("PASS: Multiple nodes retrieved (default behavior).")
            else:
                 print("NOTE: Only 1 or 0 nodes available/retrieved, hard to verify default > 1.")
        else:
            print(f"FAIL: Query failed ({resp.status_code})")

    except Exception as e:
        print(f"Verification Error: {e}")
    finally:
        print("\nStopping API...")
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait()

if __name__ == "__main__":
    verify()
