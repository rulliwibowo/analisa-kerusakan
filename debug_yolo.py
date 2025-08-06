from ultralytics import YOLO
import cv2
import os
import sys

def debug_yolo(model_path, image_path):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan: {model_path}")
        return
    if not os.path.exists(image_path):
        print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
        return

    print("[INFO] Memuat model YOLO...")
    model = YOLO(model_path)

    print("[INFO] Membaca gambar...")
    img = cv2.imread(image_path)
    if img is None:
        print("[ERROR] Gagal membaca gambar.")
        return

    img_h, img_w = img.shape[:2]
    print(f"[INFO] Ukuran gambar: {img_w}x{img_h}")

    print("[INFO] Melakukan inferensi YOLO...")
    results = model(image_path)[0]

    if len(results.boxes) == 0:
        print("[INFO] Tidak ada deteksi.")
    else:
        print(f"[INFO] Jumlah deteksi: {len(results.boxes)}")
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            print(f" - Box {i+1}: ({x1:.0f},{y1:.0f}) - ({x2:.0f},{y2:.0f}), Kelas: {model.names[cls_id]}, Confidence: {conf:.2f}")

    print("[INFO] Menyimpan hasil visualisasi ke hasil_deteksi.jpg...")
    results.save(filename="hasil_deteksi.jpg")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Cara pakai: python debug_yolo.py path_ke_model.pt path_ke_gambar.jpg")
    else:
        model_path = sys.argv[1]
        image_path = sys.argv[2]
        debug_yolo(model_path, image_path)
