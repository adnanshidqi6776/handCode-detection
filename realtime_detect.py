from ultralytics import YOLO
import torch
import warnings
import cv2

warnings.filterwarnings("ignore", category=FutureWarning)

# Muat model dan pindahkan ke device
model = YOLO(r"runs\detect\train\weights\best.pt")
model.to('cuda')

# Jalankan 1 kali dummy inference untuk inisialisasi GPU

# Buka kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Tidak dapat mengakses kamera.")
    exit()

print("Kamera aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    # Jalankan deteksi (threshold 0.8)
    results = model(frame, stream=True, conf=0.5, imgsz=640)

    for r in results:
        annotated_frame = r.plot()
        # Tambahkan teks status di pojok kiri atas
        cv2.imshow("Deteksi Objek - YOLOv8 (Realtime)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
