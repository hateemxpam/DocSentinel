"""
test_api_client.py - API Layer Verification Test
================================================

Uses FastAPI's TestClient to verify the endpoints synchronously without
spinning up a separate uvicorn process.
"""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def main():
    print("\n" + "="*50)
    print("RUNNING API ENDPOINT VERIFICATION")
    print("="*50)

    with TestClient(app) as client:
        # 1. Health Check Endpoint
        print("\n--- 1. Testing GET /health ---")
        response = client.get("/health")
        print(f"Status Code: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # 2. Stats Endpoint (Before Queries)
        print("\n--- 2. Testing GET /stats (Initial) ---")
        response = client.get("/stats")
        print(f"Status Code: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # 3. Query Endpoint
        print("\n--- 3. Testing POST /query ---")
        query_payload = {
            "query": "What is the right to lodge a complaint under GDPR?",
            "min_confidence": 0.50
        }
        response = client.post("/query", json=query_payload)
        print(f"Status Code: {response.status_code}")
        res_data = response.json()
        print(f"Status: {res_data.get('status')}")
        print(f"Latency: {res_data.get('latency_ms')} ms")
        print(f"Cached: {res_data.get('cached')}")
        print(f"Answer snippet: {res_data.get('answer')[:120]}...")
        assert response.status_code == 200

        # 4. Stats Endpoint (After Query)
        print("\n--- 4. Testing GET /stats (Updated) ---")
        response = client.get("/stats")
        print(f"Status Code: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

        # 5. Clear Cache Endpoint
        print("\n--- 5. Testing DELETE /cache ---")
        response = client.delete("/cache")
        print(f"Status Code: {response.status_code}")
        print(response.json())
        assert response.status_code == 200

    print("\n" + "="*50)
    print("ALL API ENDPOINTS TESTED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
