
# ============================================
# backend/test_api.py - Script de Prueba
# ============================================

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing /health...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}\n")

def test_analyze():
    """Test analyze endpoint"""
    print("🔍 Testing /analyze...")
    
    test_cases = [
        {
            "text": "Scientists discover breakthrough cure for cancer in groundbreaking study published in Nature.",
            "expected": "real"
        },
        {
            "text": "You won't believe what this celebrity did! Doctors hate this one weird trick to lose weight instantly!",
            "expected": "fake"
        },
        {
            "text": "The Federal Reserve announced new interest rate policies during today's press conference.",
            "expected": "real"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['text'][:60]}...")
        
        response = requests.post(
            f"{BASE_URL}/analyze",
            json={"text": test['text'], "url": "https://example.com"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Classification: {result['classification']}")
            print(f"   📊 Score: {result['score']:.1f}/100")
            print(f"   🎯 Confidence: {result['confidence']*100:.1f}%")
            print(f"   ⏱️  Time: {result['processing_time_ms']:.1f}ms")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   {response.json()}")

def test_batch():
    """Test batch analyze"""
    print("\n🔍 Testing /batch-analyze...")
    
    texts = [
        "Breaking news about politics",
        "Scientists make new discovery",
        "Click here for amazing results!"
    ]
    
    response = requests.post(f"{BASE_URL}/batch-analyze", json=texts)
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Analyzed {result['total']} texts")
        for r in result['results']:
            print(f"   - Text {r['index']}: {r.get('classification', 'error')}")
    else:
        print(f"   ❌ Error: {response.status_code}")

if __name__ == "__main__":
    print("="*60)
    print("🧪 TESTING FAKE NEWS DETECTOR API")
    print("="*60 + "\n")
    
    test_health()
    test_analyze()
    test_batch()
    
    print("\n" + "="*60)
    print("✅ Tests completados")
    print("="*60)