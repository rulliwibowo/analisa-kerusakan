from ultralytics import YOLO
import sys
import argparse
import os

# Parser untuk menerima argumen dari Flask
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=int, default=640)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--data', type=str, default='yolo/data.yaml')
parser.add_argument('--weights', type=str, default='yolov8n.pt')  # atau 'yolov8s.pt'
parser.add_argument('--name', type=str, default='yolov8_custom')
args = parser.parse_args()
args.data = os.path.abspath(args.data)

print("🚀 Mulai training YOLO dengan Ultralytics...")

# Load dan training model
model = YOLO(args.weights)
model.train(
    data=args.data,
    imgsz=args.img,
    batch=args.batch,
    epochs=args.epochs,
    name=args.name
)

print("✅ Training selesai!")
