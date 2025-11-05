import asyncio
import json
import base64
import cv2
import numpy as np
import torch
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from datetime import datetime


app = FastAPI()
app.mount("/static", StaticFiles(directory="./static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('runs/detect/train/weights/best.pt')
model.to(device)

warning_count = 0
max_warnings = None
pending_warning = False


@app.get("/")
async def get():
    return HTMLResponse("YOLO WebSocket Server Running!")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global warning_count, max_warnings, pending_warning
    await websocket.accept()
    print("✅ WS Connected")

    warning_count = 0
    max_warnings = None
    pending_warning = False

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)

            # --- Menerima konfigurasi awal ---
            if "max_warnings" in payload:
                max_warnings = int(payload["max_warnings"])
                warning_count = 0
                print(f"⚙️ Max warnings diatur ke {max_warnings}")
                continue

            # --- Menerima frame ---
            if "image" in payload:
                img_data = base64.b64decode(payload["image"].split(",")[1])
                npimg = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                # frame = cv2.flip(frame, 1)  mirror secara horizontal


                # Jalankan YOLO di thread terpisah agar tidak blocking event loop
                results = await asyncio.to_thread(model, frame)
                r = results[0]

                boxes = []
                warning_detected = False

                for box in r.boxes:
                    cls = model.names[int(box.cls)]
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if cls in ["finger", "book", "handphone"] and conf > 0.8:
                        warning_detected = True
                        print(f"Detected label={cls}, conf={conf:.2f}")
                        boxes.append({
                            "class": cls,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

                # --- Simpan frame jika ada deteksi ---
                if warning_detected:
                    save_dir = "detected_images"
                    os.makedirs(save_dir, exist_ok=True)

                    # Gambar bounding box di frame
                    annotated_frame = frame.copy()
                    for b in boxes:
                        x1, y1, x2, y2 = map(int, b["bbox"])
                        label = f"{b['class']} {b['confidence']:.2f}"
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )

                    # Simpan file dengan timestamp unik
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{save_dir}/deteksi_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"💾 Gambar disimpan: {filename}")

                # Kirim hasil bounding box ke client
                await websocket.send_text(json.dumps({
                    "boxes": boxes
                }))

                # Jika ada deteksi mencurigakan
                if warning_detected and not pending_warning:
                    pending_warning = True
                    print("🚨 Sending show_warning to frontend")
                    await websocket.send_text(json.dumps({
                        "show_warning": True,
                        "message": "⚠️ Perilaku mencurigakan terdeteksi!"
                    }))
                    await asyncio.sleep(3)

            # --- Menerima konfirmasi dari frontend ---
            if payload.get("cmd") == "ack_warning":
                if pending_warning:
                    warning_count += 1
                    pending_warning = False
                    print(f"✅ ACK diterima — warning_count = {warning_count}")

                    if max_warnings and warning_count >= max_warnings:
                        print("🚨 Stop signal dikirim ke frontend!")
                        await websocket.send_text(json.dumps({
                            "stop": True,
                            "message": f"⚠️ Deteksi kecurangan melebihi batas ({warning_count}/{max_warnings})! Kamera dihentikan."
                        }))
                        await websocket.close()
                        break

    except Exception as e:
        print("⚠️ WS Closed:", e)
    finally:
        print("❌ WS Closed")

# === Tambahkan ini agar jalan di lokal maupun Render ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)