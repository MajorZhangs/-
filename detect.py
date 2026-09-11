from ultralytics import YOLO

# 載入你訓練好的模型
model = YOLO(
    r"D:\My Code\yolov8\runs\detect\train-9\weights\best.pt"
)

# 偵測圖片test.jpg
results = model.predict(
    source=r"D:\My Code\yolov8\test.jpg",
    save=True,
    conf=0.5
)