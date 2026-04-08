from ultralytics import YOLO
import os

# 强制离线 不下载
os.environ["ULTRALYTICS_DOWNLOAD_WEIGHTS"] = "false"
os.environ["ULTRALYTICS_ENABLE_MODEL_DOWNLOAD"] = "false"

# 缓存保存到D盘
os.environ["ULTRALYTICS_CONFIG_DIR"] = r"D:\yolo_offline\config"
os.environ["ULTRALYTICS_RUNS_DIR"] = r"D:\yolo_offline\runs"

# 模型路径
MODEL_PATH = r"D:\Code\python\download\blind_road_dataset\yolov8n.pt"

# 数据集路径
DATA_YAML = r"D:\Code\python\download\blind_road_dataset\project_data\final_dataset\data.yaml"

# 保存路径
SAVE_DIR = r"D:\Code\python\download\blind_road_dataset\trained_model_v8"

if __name__ == '__main__':
    model = YOLO(MODEL_PATH, task="detect")

    model.train(
        data=DATA_YAML,
        imgsz=640,
        epochs=100,
        batch=8,
        device=0,
        workers=0,
        amp=True,
        project=SAVE_DIR,
        name="blind_road_yolov8n",
        exist_ok=True,
        patience=20,
        cache=False
    )