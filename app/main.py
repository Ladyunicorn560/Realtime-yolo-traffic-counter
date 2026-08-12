from fastapi import FastAPI, UploadFile, File, APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import os
import shutil
import asyncio
from .services.counter_service import CounterService

app = FastAPI(title="YOLOv8 Car Counter Pro")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

counter_service = CounterService()

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("app/static/index.html") as f:
        return f.read()

@app.get("/video_feed")
async def video_feed():
    async def generate():
        # Wait up to 5 seconds for the first frame to avoid broken image icons
        for _ in range(500):
            frame = counter_service.get_frame()
            if frame:
                break
            await asyncio.sleep(0.01)
            
        while True:
            frame = counter_service.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                # Keep connection alive with a tiny sleep if frame is missing
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.01)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/frame")
async def single_frame():
    """Single JPEG frame for JS polling."""
    from fastapi.responses import Response
    frame = counter_service.get_frame()
    if frame:
        return Response(content=frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache, no-store"})
    return Response(status_code=204)

@app.get("/stats")
async def get_stats():
    return counter_service.get_counts()

class StartRequest(BaseModel):
    source: str = "0"

@app.post("/start")
async def start_camera(body: StartRequest):
    """Start stream. Send JSON body: {\"source\": \"rtsp://...\"}"""
    source = body.source
    if source.isdigit():
        source = int(source)
    counter_service.start_stream(source)
    return {"status": "started", "source": source}

@app.get("/start")
async def start_camera_get(source: str = "0"):
    """Convenience GET — works for simple sources without & in URL."""
    if source.isdigit():
        source = int(source)
    counter_service.start_stream(source)
    return {"status": "started", "source": source}

@app.post("/stop")
async def stop_camera():
    counter_service.stop_stream()
    return {"status": "stopped"}

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    target_dir = "DATA/INPUTS"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    upload_path = os.path.join(target_dir, file.filename)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    counter_service.start_stream(upload_path)
    return {"status": "uploaded and started", "filename": file.filename}


# ── Webhook Configuration ─────────────────────────────────────────────────────

class WebhookConfig(BaseModel):
    url: str = ""
    api_key: Optional[str] = "my_secure_camera_token_123"

@app.post("/webhook/config")
async def set_webhook(cfg: WebhookConfig):
    """Save webhook URL (and optional API key) to the running counter service."""
    counter_service.webhook_url     = cfg.url.strip()
    counter_service.webhook_api_key = cfg.api_key or "my_secure_camera_token_123"
    status = "enabled" if counter_service.webhook_url else "disabled"
    print(f"[Webhook] {status.upper()} → {counter_service.webhook_url or '(cleared)'}")

    # Auto-ping: confirm connectivity right after saving
    # Payload matches the AnprWebhookPayload schema used by the remote server
    if counter_service.webhook_url:
        def _ping(url=counter_service.webhook_url, key=counter_service.webhook_api_key):
            import requests, json
            from datetime import datetime, timezone
            payload = {
                "camera_id": "camera_1",
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "plate": {
                    "number": "TEST-PING"
                },
                "confidence": 1.0
            }
            headers = {"X-API-Key": key, "Content-Type": "application/json"}
            print("\n" + "="*55)
            print("🔔  WEBHOOK SAVE — CONFIRMATION PING")
            print(f"🔗  URL : {url}")
            print(f"📦  Payload : {json.dumps(payload)}")
            print("="*55)
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=5)
                print(f"✅  SUCCESS | HTTP {r.status_code} | Response: {r.text}")
            except Exception as exc:
                print(f"❌  FAILED  | Could not reach {url}")
                print(f"   Error   : {exc}")
            print("="*55 + "\n", flush=True)
        import threading
        threading.Thread(target=_ping, daemon=True).start()

    return {"status": status, "url": counter_service.webhook_url}

@app.get("/webhook/config")
async def get_webhook():
    """Return current webhook settings."""
    return {
        "url": counter_service.webhook_url,
        "api_key": counter_service.webhook_api_key,
        "enabled": bool(counter_service.webhook_url)
    }

@app.post("/webhook/test")
async def test_webhook():
    """Send a test payload to the currently configured webhook URL."""
    import requests, json
    from datetime import datetime, timezone

    url = counter_service.webhook_url.strip()
    if not url:
        return {"status": "error", "message": "No webhook URL configured. Please save a URL first."}

    payload = {
        "camera_id": "test_camera",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": "vehicle_crossed",
        "vehicle": {
            "type": "car",
            "track_id": 999,
            "direction": "down"
        },
        "counts": {
            "total": 1,
            "up": 0,
            "down": 1,
            "car": 1,
            "motorbike": 0,
            "bus": 0,
            "truck": 0
        }
    }

    headers = {
        "X-API-Key": counter_service.webhook_api_key,
        "Content-Type": "application/json"
    }

    print("\n" + "="*55)
    print("🧪  WEBHOOK TEST — SENDING TO REMOTE SERVER")
    print(f"🔗  URL     : {url}")
    print(f"📦  Payload :\n{json.dumps(payload, indent=4)}")
    print("="*55)

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=5)
        print(f"✅  SUCCESS  | HTTP {response.status_code} | Response: {response.text}")
        print("="*55 + "\n", flush=True)
        return {
            "status": "success",
            "http_code": response.status_code,
            "response": response.text
        }
    except Exception as exc:
        print(f"❌  FAILED   | Error: {exc}")
        print("="*55 + "\n", flush=True)
        return {"status": "error", "message": str(exc)}

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 YOLOv8 Car Counter Pro is starting!")
    print("👉 Dashboard available at: http://127.0.0.1:8000")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
