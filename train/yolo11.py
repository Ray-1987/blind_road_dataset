from ultralytics import YOLO
import os

# ==========================================
# 🔴 强制彻底禁止下载！！！
# ==========================================
os.environ["ULTRALYTICS_DOWNLOAD_WEIGHTS"] = "false"
os.environ["ULTRALYTICS_ENABLE_MODEL_DOWNLOAD"] = "false"

# ==========================================
# 🔴 不用任何 .pt 文件！只用 YAML！
# ==========================================
if __name__ == '__main__':
    # ✅ 这一行 100% 不下载、不联网、不读损坏文件
    model = YOLO("yolo11n.yaml")

    # 训练
    model.train(
        data=r"D:\Code\python\download\blind_road_dataset\project_data\final_dataset\data.yaml",
        imgsz=640,
        epochs=50,
        batch=4,
        device=0,
        workers=0,
        cache=False,
        project=r"D:\Code\python\download\blind_road_dataset\trained",
        name="blind_road",
        exist_ok=True
    )