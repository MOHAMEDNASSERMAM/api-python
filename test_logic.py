from fastapi.testclient import TestClient
import json
from chatbotjobito import app

client = TestClient(app)

def test_chat():
    print("Testing /chat endpoint...")
    response = client.post("/chat", json={"message": "اريد وظيفة", "user_id": "test_user"})
    print(f"Status: {response.status_code}")
    print(f"Reply: {response.json()['reply']}")
    assert response.status_code == 200
    assert "وظيفة" in response.json()['reply'] or "jobs" in response.json()['reply'].lower() or "عذراً" in response.json()['reply']
    print("Chat test PASSED!")

if __name__ == "__main__":
    try:
        test_chat()
    except Exception as e:
        print(f"Test FAILED: {e}")
