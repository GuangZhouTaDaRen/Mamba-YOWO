import torch
import yaml
import os
import cv2
from models.model_entry import build_model
from utils.load_weights import load_weights
from utils.transform import video_transforms
from utils.utils import non_max_suppression
import numpy as np

# ---------- 配置部分 ----------
CONFIG_PATH = "/mnt/YOWOv3/weights/model_v3_tiny_131/config.yaml"
WEIGHTS_PATH = "/mnt/YOWOv3/weights/model_v3_tiny_131/ema_epoch_10.pth"  # 替换为你的模型路径
INPUT_VIDEO = "/mnt/YOWOv3/results/Video Video 1 2025_3_6 16_45_22 1片段[11]片段[50].mp4"  # 可替换为帧序列或单帧图像
CONF_THRESH = 0.4
IOU_THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- 加载配置 ----------
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# ---------- 加载模型 ----------
model = build_model(config)
model = model.to(DEVICE)
load_weights(model, WEIGHTS_PATH)
model.eval()

# ---------- 加载标签 ----------
idx2name = config['idx2name']

# ---------- 视频读取 ----------
cap = cv2.VideoCapture(INPUT_VIDEO)
clip_length = config['clip_length']
img_size = config['img_size']
clip = []

frame_id = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (img_size, img_size))
    clip.append(frame_resized)
    frame_id += 1

    if len(clip) < clip_length:
        continue

    # ---------- 数据预处理 ----------
    transform = video_transforms(config)
    input_clip = transform(clip)  # Tensor: (C, T, H, W)
    input_clip = input_clip.unsqueeze(0).to(DEVICE)  # (1, C, T, H, W)

    # ---------- 推理 ----------
    with torch.no_grad():
        outputs = model(input_clip)
        outputs = non_max_suppression(outputs, conf_thres=CONF_THRESH, iou_thres=IOU_THRESH)[0]

    # ---------- 可视化 ----------
    output_frame = clip[-1].copy()
    if outputs is not None:
        for det in outputs:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            label = f"{idx2name[int(cls)]} {conf:.2f}"
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(output_frame, label, (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ---------- 保存输出 ----------
    save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(save_path, output_frame)

    # 滑动窗口
    clip.pop(0)

cap.release()
print("✅ 推理完成，结果保存在:", OUTPUT_DIR)
