import asyncio
import json
import base64
import cv2
import numpy as np
import torch
import os
import csv
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

# === Fungsi untuk mencatat log pelanggaran ===
def log_violation(violation_type, confidence, filename):
    os.makedirs("logs", exist_ok=True)  # Pastikan folder logs ada
    log_file = os.path.join("logs", "violations_log.csv")  # Simpan di folder logs
    file_exists = os.path.isfile(log_file)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Pastikan file ada header kalau baru dibuat
    with open(log_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["waktu", "jenis_pelanggaran", "confidence", "file_gambar"])
        writer.writerow([timestamp, violation_type, f"{confidence:.2f}", filename])



warning_count = 0
max_warnings = None
pending_warning = False
detect_enabled = True



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
                detect_enabled = payload.get("detect_enabled", True)
                print(f"⚙️ Max warnings = {max_warnings}, Deteksi aktif = {detect_enabled}")
                continue


            # --- Menerima frame ---
            if "image" in payload:
                img_data = base64.b64decode(payload["image"].split(",")[1])
                npimg = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                # frame = cv2.flip(frame, 1)  mirror secara horizontal


                # Jalankan YOLO di thread terpisah agar tidak blocking event loop
                results = await asyncio.to_thread(model.predict, frame)
                r = results[0]

                boxes = []
                warning_detected = False

                for box in r.boxes:
                    cls = model.names[int(box.cls)]
                    conf = float(box.conf)
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    if cls in ["finger", "book", "handphone"] and conf > 0.5:
                        warning_detected = True
                        print(f"Detected label={cls}, conf={conf:.2f}")
                        boxes.append({
                            "class": cls,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2]
                        })

                
                # --- Simpan frame jika ada deteksi ---
                if warning_detected:
                    if detect_enabled:
                        # === Simpan gambar & log HANYA jika toggle ON ===
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

                        # Simpan log pelanggaran
                        log_violation(boxes[0]['class'], boxes[0]['confidence'], filename)
                    else:
                        print("⚠️ Deteksi diabaikan karena toggle OFF (tidak disimpan/log)")



                # Kirim hasil bounding box ke client
                await websocket.send_text(json.dumps({
                    "boxes": boxes
                }))

                # Jika ada deteksi mencurigakan
                if warning_detected and not pending_warning and detect_enabled:
                    pending_warning = True
                    print("🚨 Sending show_warning to frontend (toggle ON)")
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