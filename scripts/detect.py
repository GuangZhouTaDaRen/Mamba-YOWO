#
# import torch
# import torch.utils.data as data
# import torch.nn as nn
# import torchvision
# import torchvision.transforms.functional as FT
# import torch.nn.functional as F
# import torch.optim as optim
#
# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import xml.etree.ElementTree as ET
# import os
# import cv2
# import random
# import sys
# import glob
#
# from math import sqrt
#
# from cus_datasets.build_dataset import build_dataset
# from utils.box import draw_bounding_box
# from utils.box import non_max_suppression
# from model.TSN.YOWOv3 import build_yowov3
# from utils.build_config import build_config
# from utils.flops import get_info
#
# def detect(config):
#
#     #########################################################################
#     dataset = build_dataset(config, phase='test')
#     model   = build_yowov3(config)
#     get_info(config, model)
#     ##########################################################################
#     mapping = config['idx2name']
#     model.to("cuda")
#     model.eval()
#
#
#     for idx in range(dataset.__len__()):
#         origin_image, clip, bboxes, labels = dataset.__getitem__(idx, get_origin_image=True)
#         #print(bboxes)
#
#         clip = clip.unsqueeze(0).to("cuda")
#         outputs = model(clip)
#         outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]
#
#         origin_image = cv2.resize(origin_image, (config['img_size'], config['img_size']))
#
#         draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)
#
#         flag = 1
#         if flag:
#             cv2.imshow('img', origin_image)
#             k = cv2.waitKey(100)
#             if k == ord('q'):
#                 return
#         else:
#             cv2.imwrite(r"H:\detect_images\_" + str(idx) + r".jpg", origin_image)
#
#             print("ok")
#             print("image {} saved!".format(idx))
#
# if __name__ == "__main__":
#     config = build_config()
#     detect(config)

import torch
import cv2
import os
from utils.box import draw_bounding_box
from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3

def detect_video(config, video_path):
    # 构建模型
    model = build_yowov3(config)
    model.load_state_dict(torch.load(config['pretrain_path']))  # 加载训练好的权重
    model.to("cuda")
    model.eval()

    # 获取类别映射字典
    mapping = config['idx2name']

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    # 创建保存结果视频的目录
    save_dir = config.get('save_dir', 'results')
    os.makedirs(save_dir, exist_ok=True)

    # 获取视频的基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 保存视频输出路径
    output_video_path = os.path.join(save_dir, "output_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    while True:
        # 读取一帧视频
        ret, frame = cap.read()
        if not ret:
            break  # 如果没有读取到帧，结束

        # 预处理输入帧
        origin_image = frame
        frame_resized = cv2.resize(frame, (config['img_size'], config['img_size']))  # 将帧大小调整为模型输入大小
        clip = torch.tensor(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to('cuda')  # 转换为 tensor 并加到 GPU 上

        # 模型前向推理
        outputs = model(clip)
        outputs = non_max_suppression(outputs, conf_threshold=0.3, iou_threshold=0.5)[0]

        # 绘制检测框
        draw_bounding_box(origin_image, outputs[:, :4], outputs[:, 5], outputs[:, 4], mapping)

        # 将带有检测框的帧写入输出视频
        out.write(origin_image)

        # 可选：在窗口中显示帧（调试时使用）
        cv2.imshow('Detection Result', origin_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 退出
            break

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"[INFO] Saved output video to {output_video_path}")

