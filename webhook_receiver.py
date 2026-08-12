"""
Vehicle Counter — Webhook Test Receiver
=======================================
Run this to verify that vehicle-crossing events are being sent correctly.

    python webhook_receiver.py

Then in the UI, set the webhook URL to:
    http://127.0.0.1:9000/webhook

Every time a vehicle crosses the counting line you'll see the full payload
printed here in the console.
"""

from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/webhook")
async def receive_webhook(request: Request):
    data    = await request.json()
    api_key = request.headers.get("x-api-key")
    print("\n" + "="*55)
    print("🚀 📥 VEHICLE COUNTER — WEBHOOK EVENT RECEIVED:")
    print(f"🔑 API Key : {api_key}")
    print("="*55)
    print(json.dumps(data, indent=4))
    print("="*55 + "\n")
    return {"status": "success", "message": "Webhook received successfully!"}

if __name__ == "__main__":
    import uvicorn
    print("\nStarting Vehicle Counter Webhook Receiver on port 9000...")
    print("Point the UI webhook URL to: http://127.0.0.1:9000/webhook\n")
    uvicorn.run(app, host="0.0.0.0", port=9000)
