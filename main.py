from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"D:\My Code\yolov8\dataset\data.yaml",
        epochs=100,
        #完整學習次數
        imgsz=640,
        #調整圖片大小
        batch=16
        #一次訓練所需圖片量
    )

if __name__ == "__main__":
    main()