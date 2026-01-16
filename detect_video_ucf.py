# import os
# import cv2
# import torch
# import yaml
# import numpy as np
# from collections import deque
# from typing import Union, Tuple, List, Dict
#
# from utils.box import non_max_suppression
# from model.TSN.YOWOv3 import build_yowov3
#
#
# # -------------------- 与 UCF_transform 完全一致的 img_size 解析 -------------------- #
# def _parse_img_size(img_size: Union[int, Tuple[int, int], List[int], Dict]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
#     if isinstance(img_size, int):
#         h = w = int(img_size)
#         return (h, w), (w, h)
#     if isinstance(img_size, (list, tuple)):
#         assert len(img_size) == 2, f"img_size 长度应为2，当前为 {img_size}"
#         h, w = int(img_size[0]), int(img_size[1])
#         return (h, w), (w, h)
#     if isinstance(img_size, dict):
#         if 'h' in img_size and 'w' in img_size:
#             h, w = int(img_size['h']), int(img_size['w'])
#             return (h, w), (w, h)
#         if 'height' in img_size and 'width' in img_size:
#             h, w = int(img_size['height']), int(img_size['width'])
#             return (h, w), (w, h)
#         raise ValueError(f"无法解析的 img_size 字段：{img_size}")
#     raise TypeError(f"img_size 类型不支持：{type(img_size)}")
#
#
# # -------------------- 加载训练时的配置 -------------------- #
# def load_config():
#     """
#     直接从你训练用的 yaml 加载 config
#     """
#     config_path = "/data/CuiTengPeng/YOWOv3/config/cf/ucf_mamba_Channel_clip12.yaml"
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     return config
#
#
# # -------------------- 构建并加载权重 -------------------- #
# def build_model(config):
#     model = build_yowov3(config)
#
#     weight_path = config['pretrain_path']
#     if not os.path.exists(weight_path):
#         raise FileNotFoundError(f"Weight file not found: {weight_path}")
#
#     print(f"[INFO] Loading weights from: {weight_path}")
#
#     state = torch.load(weight_path, map_location="cpu")
#     # 兼容 ckpt 里带 'model_state_dict' 的情况
#     if isinstance(state, dict) and 'model_state_dict' in state:
#         state = state['model_state_dict']
#
#     model.load_state_dict(state, strict=False)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#
#     return model, device
#
#
# # -------------------- 自定义画框（加粗 & 字体黑色） -------------------- #
# def draw_boxes_thick(image, boxes, clses, confs, mapping):
#     """
#     在 image 上画检测框：
#       - 框线加粗
#       - 字体稍大
#       - 字体颜色为黑色，绿色底框
#     boxes, clses, confs 都是 torch.Tensor（CPU）
#     """
#     box_color = (0, 255, 0)           # 绿色框/背景
#     box_thickness = 4
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0                  # 大约是默认 0.5 的 2 倍
#     font_thickness = 2                # 字体稍粗
#
#     h, w = image.shape[:2]
#
#     for box, cls_id, conf in zip(boxes, clses, confs):
#         x1, y1, x2, y2 = box.tolist()
#         x1 = int(max(0, min(w - 1, x1)))
#         y1 = int(max(0, min(h - 1, y1)))
#         x2 = int(max(0, min(w - 1, x2)))
#         y2 = int(max(0, min(h - 1, y2)))
#
#         # 类别名称
#         cls_idx = int(cls_id.item()) if isinstance(cls_id, torch.Tensor) else int(cls_id)
#         class_name = mapping.get(cls_idx, str(cls_idx))
#
#         # 文本内容：类别名 + 置信度
#         score = float(conf.item()) if isinstance(conf, torch.Tensor) else float(conf)
#         label = f"{class_name}:{score:.2f}"
#
#         # 画矩形框
#         cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness=box_thickness)
#
#         # 文字位置
#         (text_width, text_height), baseline = cv2.getTextSize(
#             label, font, font_scale, font_thickness
#         )
#         text_x = x1
#         text_y = y1 - 5
#
#         if text_y - text_height - baseline < 0:
#             text_y = y1 + text_height + baseline + 5
#
#         # 背景框（绿色）
#         cv2.rectangle(
#             image,
#             (text_x, text_y - text_height - baseline),
#             (text_x + text_width, text_y + baseline),
#             box_color,
#             thickness=-1,
#         )
#
#         # 黑色文字
#         cv2.putText(
#             image,
#             label,
#             (text_x, text_y),
#             font,
#             font_scale,
#             (0, 0, 0),           # 字体黑色
#             thickness=font_thickness,
#             lineType=cv2.LINE_AA,
#         )
#
#
# # -------------------- 生成检测热力图 -------------------- #
# def build_detection_heatmap(height, width, boxes, confs):
#     """
#     简单的“检测热力图”：
#       - 对每个框区域累加其置信度
#       - 再归一化到 [0, 255]，用 COLORMAP_JET 上色
#     """
#     heatmap = np.zeros((height, width), dtype=np.float32)
#
#     if boxes is None or len(boxes) == 0:
#         return None
#
#     boxes_np = boxes.numpy()
#     confs_np = confs.numpy()
#
#     for box, conf in zip(boxes_np, confs_np):
#         x1, y1, x2, y2 = box
#         x1 = int(max(0, min(width - 1, x1)))
#         y1 = int(max(0, min(height - 1, y1)))
#         x2 = int(max(0, min(width - 1, x2)))
#         y2 = int(max(0, min(height - 1, y2)))
#         if x2 > x1 and y2 > y1:
#             heatmap[y1:y2, x1:x2] += float(conf)
#
#     if heatmap.max() <= 0:
#         return None
#
#     heatmap_norm = (heatmap / heatmap.max() * 255).astype(np.uint8)
#     heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
#     return heatmap_color
#
#
# # -------------------- 视频预测 -------------------- #
# @torch.no_grad()
# def detect_video(config, video_path):
#     # 1. 模型 & 设备
#     model, device = build_model(config)
#
#     # 2. 类别映射
#     mapping = config['idx2name']
#
#     # 3. 打开视频
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"[ERROR] Unable to open video: {video_path}")
#         return
#
#     # 4. 输出目录
#     save_dir = config.get('save_dir', 'results_video_ucf')
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 带框+热力图帧目录
#     frame_save_dir = os.path.join(save_dir, "frames")
#     os.makedirs(frame_save_dir, exist_ok=True)
#
#     # 原始帧目录
#     raw_frame_save_dir = os.path.join(save_dir, "frames_raw")
#     os.makedirs(raw_frame_save_dir, exist_ok=True)
#
#     # 5. 视频信息
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 25.0
#     orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"[INFO] Video FPS: {fps}, size: {orig_width}x{orig_height}")
#
#     # 输出视频（带框 + 热力图）
#     output_video_path = os.path.join(save_dir, "output_video.mp4")
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))
#
#     # 6. 输入尺寸 & clip 长度（严格对齐 UCF_transform）
#     size_hw, size_wh = _parse_img_size(config['img_size'])  # size_hw = (H, W), size_wh = (W, H)
#     H_in, W_in = size_hw
#     clip_len = config.get('clip_len', 12)
#
#     frame_buffer = deque(maxlen=clip_len)
#
#     # 预先准备 mean/std，用于标准化 [C, T, H, W]
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
#     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
#
#     frame_idx = 0
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         frame_idx += 1
#
#         # 保存原始帧（无框）
#         raw_frame_path = os.path.join(raw_frame_save_dir, f"raw_{frame_idx:06d}.jpg")
#         cv2.imwrite(raw_frame_path, frame)
#
#         origin_image = frame.copy()  # 用于后续叠加热力图和画框
#
#         # ---------- 预处理：对齐 UCF_transform 的 resize + to_tensor ---------- #
#         # resize 到 (W, H) = size_wh
#         resized = cv2.resize(origin_image, size_wh)  # size_wh: (W, H)
#         rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
#         # HWC -> CHW，/255.0（等价于 F.to_tensor）
#         frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3, H_in, W_in]
#
#         frame_buffer.append(frame_tensor)
#
#         # 缓存不足 clip_len 时，仅原样输出
#         if len(frame_buffer) < clip_len:
#             # 没有模型输出，就没有热力图，直接保存原图
#             out.write(origin_image)
#             cv2.imwrite(os.path.join(frame_save_dir, f"frame_{frame_idx:06d}.jpg"), origin_image)
#             continue
#
#         # ---------- 构造 [C, T, H, W] 并标准化（完全照搬 UCF_transform._normalize） ---------- #
#         clip_cthw = torch.stack(list(frame_buffer), dim=1)  # [3, T, H, W]
#
#         # 标准化： (clip - mean) / std
#         clip_norm = (clip_cthw - mean) / std  # mean/std 在 CPU 上，和训练/测试一致
#
#         # 扩 batch 维度：[1, 3, T, H, W]
#         clips = clip_norm.unsqueeze(0).to(device)
#
#         # ---------- 模型前向 ---------- #
#         outputs = model(clips)
#
#         # 和 eval 中一致的阈值：conf=0.5, iou=0.5
#         nms_outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)
#
#         detections = None
#         if nms_outputs is not None and len(nms_outputs) > 0:
#             detections = nms_outputs[0]
#
#         # 先准备一个 overlay，用于叠加热力图 & 画框
#         overlay = origin_image.copy()
#
#         if detections is not None and len(detections) > 0:
#             detections = detections.detach().cpu()
#             boxes = detections[:, :4].clone()
#             confs = detections[:, 4].clone()
#             clses = detections[:, 5].clone()
#
#             # 模型是在 (W_in, H_in) 尺度上预测的，映射回原始分辨率
#             ratio_w = orig_width / float(W_in)
#             ratio_h = orig_height / float(H_in)
#             boxes[:, [0, 2]] *= ratio_w
#             boxes[:, [1, 3]] *= ratio_h
#
#             # ---------- 生成检测热力图并叠加 ---------- #
#             heatmap_color = build_detection_heatmap(orig_height, orig_width, boxes, confs)
#             if heatmap_color is not None:
#                 # 0.6 * 原图 + 0.4 * 热力图
#                 overlay = cv2.addWeighted(overlay, 0.6, heatmap_color, 0.4, 0)
#
#             # ---------- 在叠加后的图上画框和类别 ---------- #
#             draw_boxes_thick(overlay, boxes, clses, confs, mapping)
#
#         # ---------- 写视频 & 保存带框+热力图帧 ---------- #
#         out.write(overlay)
#         cv2.imwrite(os.path.join(frame_save_dir, f"frame_{frame_idx:06d}.jpg"), overlay)
#
#     cap.release()
#     out.release()
#
#     print(f"[INFO] Saved output video to: {output_video_path}")
#     print(f"[INFO] Saved raw frames to: {raw_frame_save_dir}")
#     print(f"[INFO] Saved boxed+heatmap frames to: {frame_save_dir}")
#
#
# # -------------------- 主入口 -------------------- #
# if __name__ == "__main__":
#     # 1. 加载训练用 config
#     config = load_config()
#
#     # 2. 指定 EMA 权重
#     config['pretrain_path'] = (
#         "/data/CuiTengPeng/YOWOv3/weights/fig/ucf/"
#         "mambaOut_shufflenetv2+Channel_clip12/ema_epoch_10.pth"
#     )
#
#     # 3. 结果保存目录
#     config['save_dir'] = "results_ucf_mambaOut"
#
#     # 4. 待检测视频路径
#     video_path = "/data/CuiTengPeng/YOWOv3/feeding_no_288.mp4"
#
#     detect_video(config, video_path)


import os
import cv2
import torch
import yaml
import numpy as np
from typing import Union, Tuple, List, Dict

from utils.box import non_max_suppression
from model.TSN.YOWOv3 import build_yowov3


# -------------------- 与 UCF_transform 完全一致的 img_size 解析 -------------------- #
def _parse_img_size(img_size: Union[int, Tuple[int, int], List[int], Dict]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    返回:
        size_hw: (H, W)
        size_wh: (W, H)
    """
    if isinstance(img_size, int):
        h = w = int(img_size)
        return (h, w), (w, h)
    if isinstance(img_size, (list, tuple)):
        assert len(img_size) == 2, f"img_size 长度应为2，当前为 {img_size}"
        h, w = int(img_size[0]), int(img_size[1])
        return (h, w), (w, h)
    if isinstance(img_size, dict):
        if 'h' in img_size and 'w' in img_size:
            h, w = int(img_size['h']), int(img_size['w'])
            return (h, w), (w, h)
        if 'height' in img_size and 'width' in img_size:
            h, w = int(img_size['height']), int(img_size['width'])
            return (h, w), (w, h)
        raise ValueError(f"无法解析的 img_size 字段：{img_size}")
    raise TypeError(f"img_size 类型不支持：{type(img_size)}")


# -------------------- 加载训练时的配置 -------------------- #
def load_config():
    """
    直接从你训练用的 yaml 加载 config
    """
    config_path = "/data/CuiTengPeng/YOWOv3/config/cf/ucf_mamba_Channel.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


# -------------------- 构建并加载权重 -------------------- #
def build_model(config):
    model = build_yowov3(config)

    weight_path = config['pretrain_path']
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    print(f"[INFO] Loading weights from: {weight_path}")

    state = torch.load(weight_path, map_location="cpu")
    # 兼容 ckpt 里带 'model_state_dict' 的情况
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']

    model.load_state_dict(state, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return model, device


# -------------------- 自定义画框（加粗 & 字体黑色） -------------------- #
def draw_boxes_thick(image, boxes, clses, confs, mapping):
    """
    在 image 上画检测框：
      - 框线加粗
      - 字体稍大
      - 字体颜色为黑色，绿色底框
    boxes, clses, confs 都是 torch.Tensor（CPU）
    """
    box_color = (0, 255, 0)           # 绿色框/背景
    box_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0                  # 大约是默认 0.5 的 2 倍
    font_thickness = 2                # 字体稍粗

    h, w = image.shape[:2]

    for box, cls_id, conf in zip(boxes, clses, confs):
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        # 类别名称
        cls_idx = int(cls_id.item()) if isinstance(cls_id, torch.Tensor) else int(cls_id)
        class_name = mapping.get(cls_idx, str(cls_idx))

        # 文本内容：类别名 + 置信度
        score = float(conf.item()) if isinstance(conf, torch.Tensor) else float(conf)
        label = f"{class_name}:{score:.2f}"

        # 画矩形框
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness=box_thickness)

        # 文字位置
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        text_x = x1
        text_y = y1 - 5

        if text_y - text_height - baseline < 0:
            text_y = y1 + text_height + baseline + 5

        # 背景框（绿色）
        cv2.rectangle(
            image,
            (text_x, text_y - text_height - baseline),
            (text_x + text_width, text_y + baseline),
            box_color,
            thickness=-1,
        )

        # 黑色文字
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),           # 字体黑色
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )


# -------------------- 按“图片形式”对文件夹下图片做检测（无热力图） -------------------- #
@torch.no_grad()
def detect_images_in_folder(config, folder_path):
    """
    对 folder_path 下的所有图片逐张做检测：
      - 每张图片作为一个 clip，通过“复制同一帧 clip_len 次”来适配 [1, 3, T, H, W] 输入
      - 输出：原图 + 画框检测结果（无热力图）
    """
    # 1. 模型 & 设备
    model, device = build_model(config)

    # 2. 类别映射
    mapping = config['idx2name']

    # 3. 输出目录
    save_dir = config.get('save_dir', 'results_ucf_mambaOut_images')
    os.makedirs(save_dir, exist_ok=True)

    # 子文件夹保存结果
    raw_dir = os.path.join(save_dir, "raw")       # 原始图片
    boxed_dir = os.path.join(save_dir, "boxed")   # 带框图片
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(boxed_dir, exist_ok=True)

    # 4. 输入图像路径列表
    image_files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if not image_files:
        print(f"[ERROR] No images found in: {folder_path}")
        return

    # 5. 输入尺寸 & clip 长度（与训练保持一致）
    size_hw, size_wh = _parse_img_size(config['img_size'])  # size_hw = (H, W), size_wh = (W, H)
    H_in, W_in = size_hw
    clip_len = config.get('clip_len', 12)

    # mean/std：用于标准化 [C, T, H, W]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)

    for img_path in image_files:
        img_name = os.path.basename(img_path)

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[WARN] Failed to read image: {img_path}")
            continue

        orig_h, orig_w = frame.shape[:2]
        origin_image = frame.copy()

        # 保存原图（用原文件名）
        raw_save_path = os.path.join(raw_dir, img_name)
        cv2.imwrite(raw_save_path, origin_image)

        # ---------- 预处理：resize + RGB + to_tensor ---------- #
        resized = cv2.resize(origin_image, size_wh)  # size_wh: (W, H)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [3, H_in, W_in]

        # ---------- 构造一个“伪 clip”：同一帧复制 clip_len 次 ---------- #
        clip_cthw = frame_tensor.unsqueeze(1).repeat(1, clip_len, 1, 1)  # [3, T, H, W]

        # 标准化 (与 UCF_transform._normalize 一致)
        clip_norm = (clip_cthw - mean) / std    # 仍在 CPU 上
        clips = clip_norm.unsqueeze(0).to(device)   # [1, 3, T, H, W]

        # ---------- 模型前向 ---------- #
        outputs = model(clips)

        # 与视频 eval 中一致的阈值
        nms_outputs = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)
        detections = None
        if nms_outputs is not None and len(nms_outputs) > 0:
            detections = nms_outputs[0]

        # 结果图：原图 + 框（如果有检测）
        result_image = origin_image.copy()

        if detections is not None and len(detections) > 0:
            detections = detections.detach().cpu()
            boxes = detections[:, :4].clone()
            confs = detections[:, 4].clone()
            clses = detections[:, 5].clone()

            # 模型是在 (W_in, H_in) 尺度上预测的，映射回原始分辨率
            ratio_w = orig_w / float(W_in)
            ratio_h = orig_h / float(H_in)
            boxes[:, [0, 2]] *= ratio_w
            boxes[:, [1, 3]] *= ratio_h

            # 在原图上画框（无热力图）
            draw_boxes_thick(result_image, boxes, clses, confs, mapping)

        # 保存带框结果（前面加个前缀，方便区分）
        boxed_save_path = os.path.join(boxed_dir, f"det_{img_name}")
        cv2.imwrite(boxed_save_path, result_image)
        print(f"[INFO] Saved detection result: {boxed_save_path}")

    print(f"[INFO] All raw images saved to: {raw_dir}")
    print(f"[INFO] All detection results saved to: {boxed_dir}")


# -------------------- 主入口 -------------------- #
if __name__ == "__main__":
    # 1. 加载训练用 config
    config = load_config()

    # 2. 指定 EMA 权重
    config['pretrain_path'] = (
        "/data/CuiTengPeng/YOWOv3/weights/fig/ucf/mambaOut_shufflenetv2+Channel/ema_epoch_10.pth"
    )

    # 3. 结果保存目录
    config['save_dir'] = "results_ucf_mambaOut_images_10"

    # 4. 待检测图片文件夹路径（你给的 feeding_no3_8471）
    folder_path = "/data/CuiTengPeng/YOWOv3/cus_datasets/ucf24/rgb-images/flying/flying_no_7411"

    detect_images_in_folder(config, folder_path)
