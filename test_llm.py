from app.llm.client import llm

resp = llm.complete_json('Return JSON: {"status": "ok", "number": 42}')
print("Response:", resp)
assert resp.get("status") == "ok", "LLM did not return expected JSON"
print("✅ LLM works")