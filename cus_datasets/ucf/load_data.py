# import torch
# import torch.utils.data as data
# import argparse
# import yaml
# import os
# import cv2
# import pickle
# import numpy as np
# from .transforms import Augmentation, UCF_transform
# from PIL import Image
#
#
# class UCF_dataset(data.Dataset):
#
#     def __init__(self, root_path, split_path, data_path, ann_path, clip_length, sampling_rate, img_size, phase,
#                  transform=Augmentation()):
#         self.root_path = root_path  # path to root folder
#         self.split_path = os.path.join(root_path, split_path)  # path to split file
#         self.data_path = os.path.join(root_path, data_path)  # path to data folder
#         self.ann_path = os.path.join(root_path, ann_path)  # path to annotation foler
#         self.transform = transform
#         self.clip_length = clip_length
#         self.sampling_rate = sampling_rate
#         self.phase = phase
#
#         with open(self.split_path, 'r') as f:
#             self.lines = f.readlines()
#
#         self.nSample = len(self.lines)
#         self.img_size = img_size
#
#     def __len__(self):
#         return self.nSample
#
#     def __getitem__(self, index, get_origin_image=False):
#         key_frame_path = self.lines[index].rstrip()  # e.g : labels/Basketball/v_Basketball_g08_c01/00070.txt
#         # for linux, replace '/' by '\' for window
#         split_parts = key_frame_path.split('/')  # e.g : ['labels', 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
#         key_frame_idx = int(split_parts[-1].split('.')[-2])  # e.g : 70
#         video_name = split_parts[-2]  # e.g : v_Basketball_g08_c01
#         class_name = split_parts[1]  # e.g : Baseketball
#         video_path = os.path.join(self.data_path, class_name, video_name)
#         ann_path = os.path.join(self.ann_path, class_name, video_name)
#         # e.g : /home/manh/Datasets/UCF101-24/ucf24/rgb-images/Basketball/v_Basketball_g08_c01
#
#         path = os.path.join(class_name, video_name)  # e.g : Basketball/v_Basketball_g08_c01
#         clip = []
#         boxes = []
#         for i in reversed(range(self.clip_length)):
#             cur_frame_idx = key_frame_idx - i * self.sampling_rate
#
#             if cur_frame_idx < 1:
#                 cur_frame_idx = 1
#
#             # get frame
#             cur_frame_path = os.path.join(video_path, '{:05d}.jpg'.format(cur_frame_idx))
#             # cur_frame      = cv2.imread(cur_frame_path)/255.0
#             # H, W, C        = cur_frame.shape
#             # cur_frame      = cv2.resize(cur_frame, self.img_size)
#             # clip.append(cur_frame)
#             cur_frame = Image.open(cur_frame_path).convert('RGB')
#             clip.append(cur_frame)
#
#         if get_origin_image == True:
#             key_frame_path = os.path.join(video_path, '{:05d}.jpg'.format(key_frame_idx))
#             original_image = cv2.imread(key_frame_path)
#
#         # get annotation for key frame
#         ann_file_name = os.path.join(ann_path, '{:05d}.txt'.format(key_frame_idx))
#         boxes = []
#         labels = []
#
#         with open(ann_file_name) as f:
#             lines = f.readlines()
#             for i, line in enumerate(lines):
#                 line = line.rstrip().split(' ')
#                 label = int(line[0])
#
#                 if self.phase == 'train':
#                     onehot_vector = np.zeros(4)
#                     onehot_vector[label] = 1.
#                     labels.append(onehot_vector)
#                 elif self.phase == 'test':
#                     labels.append(label)
#
#                 box = [float(line[1]), float(line[2]), float(line[3]), float(line[4])]
#                 boxes.append(box)
#
#         boxes = np.array(boxes)
#
#         if self.phase == 'train':
#             labels = np.array(labels)
#         elif self.phase == 'test':
#             labels = np.expand_dims(np.array(labels), axis=1)
#
#         targets = np.concatenate((boxes, labels), axis=1)
#         clip, targets = self.transform(clip, targets)
#
#         boxes = targets[:, :4]
#
#         if self.phase == 'train':
#             labels = targets[:, 4:]
#         elif self.phase == 'test':
#             labels = targets[:, -1]
#
#         if get_origin_image == True:
#             return original_image, clip, boxes, labels
#         else:
#             return clip, boxes, labels
#
#
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#
#     load_dotenv()
#
#     root_path = os.getenv("YOWO_UCF_ROOT_PATH")
#     split_path = "trainlist.txt"
#     data_path = "rgb-images"
#     ann_path = "labels"
#     clip_length = 16
#     sampling_rate = 1
#
#     dataset = UCF_dataset(root_path, split_path, data_path, ann_path
#                           , clip_length, sampling_rate, img_size=(224, 224))
#
#     for i in range(13000, dataset.__len__()):
#         original_image, clip, boxes, labels = dataset.__getitem__(i, get_origin_image=True)
#         original_image = clip[:, -1, :, :].squeeze(1).permute(1, 2, 0).contiguous().numpy()
#
#         for box, label in zip(boxes, labels):
#             H, W, C = original_image.shape
#
#             pt1 = (int(box[0] * W), int(box[1] * H))
#             pt2 = (int(box[2] * W), int(box[3] * H))
#
#             cv2.rectangle(original_image, pt1, pt2, 1, 1, 1)
#             print(label)
#
#         cv2.imshow('img', original_image)
#         k = cv2.waitKey()
#         if k == ord('q'):
#             break
#
#
# def build_ucf_dataset(config, phase):
#     root_path = config['data_root']
#     data_path = "rgb-images"
#     ann_path = "labels"
#     clip_length = config['clip_length']
#     sampling_rate = config['sampling_rate']
#     img_size = config['img_size']
#
#     if phase == 'train':
#         split_path = "trainlist.txt"
#         return UCF_dataset(root_path, split_path, data_path, ann_path
#                            , clip_length, sampling_rate, img_size, transform=Augmentation(img_size=img_size),
#                            phase=phase)
#     elif phase == 'test':
#         split_path = "testlist.txt"
#         return UCF_dataset(root_path, split_path, data_path, ann_path
#                            , clip_length, sampling_rate, img_size, transform=UCF_transform(img_size=img_size),
#                            phase=phase)

import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from .transforms import Augmentation, UCF_transform


class UCF_dataset(data.Dataset):
    def __init__(self,
                 root_path,
                 split_path,
                 data_path,
                 ann_path,
                 clip_length,
                 sampling_rate,
                 img_size,
                 phase,
                 num_classes,
                 transform=None):
        """
        Args:
            root_path: 数据根目录
            split_path: 划分文件（trainlist.txt / testlist.txt）
            data_path: 图像目录（相对 root_path 的子目录），默认 "rgb-images"
            ann_path: 标注目录（相对 root_path 的子目录），默认 "labels"
            clip_length: 每个样本的帧数
            sampling_rate: 采样间隔
            img_size: (H, W) 元组，用于变换
            phase: "train" or "test"
            num_classes: 类别数（训练阶段用于 one-hot）
            transform: 数据增强/变换函数（可为 None，外部 build 函数会按 phase 设置）
        """
        assert phase in ['train', 'test'], "phase 必须是 'train' 或 'test'"

        self.root_path = root_path
        self.split_path = os.path.join(root_path, split_path)
        self.data_path = os.path.join(root_path, data_path)
        self.ann_path = os.path.join(root_path, ann_path)
        self.clip_length = int(clip_length)
        self.sampling_rate = int(sampling_rate)
        self.img_size = tuple(img_size)
        self.phase = phase
        self.num_classes = int(num_classes)
        self.transform = transform

        if not os.path.isfile(self.split_path):
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        with open(self.split_path, 'r') as f:
            # 过滤空行
            self.lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        self.nSample = len(self.lines)

    def __len__(self):
        return self.nSample

    def __getitem__(self, index, get_origin_image=False):
        """
        返回:
            (clip, boxes, labels)
            或 (original_image, clip, boxes, labels) 当 get_origin_image=True
        其中：
            clip: 经过 transform 的时序图像张量 [C, T, H, W]
            boxes: (N, 4) numpy，通常为相对坐标（由 transform 决定）
            labels:
                train: (N, C) one-hot 的 numpy
                test : (N,) 的类别 id numpy（若 targets 为一热，会做 argmax）
        """
        # 例：labels/Basketball/v_Basketball_g08_c01/00070.txt
        key_frame_rel = self.lines[index]
        parts = key_frame_rel.split('/')
        if len(parts) < 4:
            raise ValueError(f"split 行格式异常：{key_frame_rel}")

        # 解析关键帧索引、类别与视频名
        try:
            key_frame_idx = int(parts[-1].split('.')[-2])  # e.g. 70
        except Exception as e:
            raise ValueError(f"无法解析关键帧编号：{parts[-1]} ({e})")

        video_name = parts[-2]      # e.g. v_Basketball_g08_c01
        class_name = parts[1]       # e.g. Basketball

        video_path = os.path.join(self.data_path, class_name, video_name)
        ann_dir = os.path.join(self.ann_path, class_name, video_name)

        # 读取 clip
        clip = []
        for i in reversed(range(self.clip_length)):
            cur_idx = key_frame_idx - i * self.sampling_rate
            if cur_idx < 1:
                cur_idx = 1
            img_p = os.path.join(video_path, f'{cur_idx:05d}.jpg')
            if not os.path.isfile(img_p):
                raise FileNotFoundError(f'Frame not found: {img_p}')
            img = Image.open(img_p).convert('RGB')
            clip.append(img)

        if get_origin_image:
            key_img_path = os.path.join(video_path, f'{key_frame_idx:05d}.jpg')
            original_image = cv2.imread(key_img_path)

        # 读取关键帧标注
        ann_file = os.path.join(ann_dir, f'{key_frame_idx:05d}.txt')
        boxes_list, labels_list = [], []

        if os.path.isfile(ann_file):
            with open(ann_file, 'r') as f:
                ann_lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for ln in ann_lines:
                seg = ln.split()
                # 期望格式: <class_id> x1 y1 x2 y2   (像素坐标/或相对坐标按你的标注而定)
                if len(seg) < 5:
                    # 跳过异常行
                    continue
                try:
                    label_id = int(float(seg[0]))
                    x1 = float(seg[1]); y1 = float(seg[2])
                    x2 = float(seg[3]); y2 = float(seg[4])
                except Exception:
                    # 跳过不可解析的行
                    continue

                boxes_list.append([x1, y1, x2, y2])
                if self.phase == 'train':
                    # 训练阶段严格检查标签范围，越界则跳过该框
                    if not (0 <= label_id < self.num_classes):
                        boxes_list.pop()
                        continue
                    oh = np.zeros(self.num_classes, dtype=np.float32)
                    oh[label_id] = 1.0
                    labels_list.append(oh)
                else:
                    labels_list.append(int(label_id))

        # ---- 标注规范化，避免维度问题 ----
        boxes = np.asarray(boxes_list, dtype=np.float32)
        if self.phase == 'train':
            labels = np.asarray(labels_list, dtype=np.float32)  # (N, C)
        else:
            labels = np.asarray(labels_list, dtype=np.int64)    # (N,)

        # boxes -> (N,4)
        if boxes.size == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
        else:
            if boxes.ndim == 1:
                if boxes.size == 4:
                    boxes = boxes.reshape(1, 4)
                else:
                    raise ValueError(f"boxes 形状异常: {boxes.shape}, 内容={boxes}")
            elif boxes.ndim == 2:
                if boxes.shape[1] != 4:
                    raise ValueError(f"boxes 每行应为4个数，当前形状: {boxes.shape}")
            else:
                raise ValueError(f"boxes 维度异常: {boxes.ndim}")

        # labels -> train:(N,C)  test:(N,1)
        if self.phase == 'train':
            if labels.size == 0:
                labels = np.zeros((0, self.num_classes), dtype=np.float32)
            else:
                if labels.ndim == 1:
                    if labels.size == self.num_classes:
                        labels = labels.reshape(1, self.num_classes)
                    else:
                        ids = labels.astype(np.int64).reshape(-1)
                        ids = np.clip(ids, 0, self.num_classes - 1)
                        labels = np.eye(self.num_classes, dtype=np.float32)[ids]
                elif labels.ndim == 2:
                    if labels.shape[1] != self.num_classes:
                        if labels.shape[1] == 1:
                            ids = labels.reshape(-1).astype(np.int64)
                            ids = np.clip(ids, 0, self.num_classes - 1)
                            labels = np.eye(self.num_classes, dtype=np.float32)[ids]
                        else:
                            raise ValueError(
                                f"train labels 列数应为 num_classes={self.num_classes}，当前 {labels.shape}"
                            )
                else:
                    raise ValueError(f"train labels 维度异常: {labels.ndim}")
        else:
            if labels.size == 0:
                labels = np.zeros((0, 1), dtype=np.int64)
            else:
                if labels.ndim == 0:
                    labels = labels.reshape(1, 1)
                elif labels.ndim == 1:
                    labels = labels.reshape(-1, 1)
                elif labels.ndim == 2:
                    # 若已有多列（一热），保留
                    pass
                else:
                    raise ValueError(f"test labels 维度异常: {labels.ndim}")

        # 对齐 N
        n_box, n_lab = boxes.shape[0], labels.shape[0]
        if n_box != n_lab:
            if self.phase == 'test' and n_lab == 1 and n_box > 1:
                labels = np.repeat(labels, n_box, axis=0)
            elif self.phase == 'train' and labels.ndim == 2 and labels.shape[0] == 1 and n_box > 1:
                labels = np.repeat(labels, n_box, axis=0)
            else:
                n = min(n_box, n_lab)
                boxes = boxes[:n]
                labels = labels[:n]

        # 组合 targets: (N, 4 + C) 或 (N, 5)（测试阶段若 labels 已是一热则为 4+C）
        targets = np.concatenate((boxes, labels.astype(np.float32)), axis=1)

        # 变换（增强 / 归一化等）—— 注意：transform 可能返回 Tensor 类型的 targets
        if self.transform is not None:
            clip, targets = self.transform(clip, targets)

        # ---- 统一将 targets 转为 NumPy，避免对 Tensor 做 astype 报错 ----
        if isinstance(targets, torch.Tensor):
            targets_np = targets.detach().cpu().numpy()
        else:
            targets_np = np.asarray(targets, dtype=np.float32)

        # 拆回 boxes / labels（保持 NumPy，和你的评估/拼接逻辑一致）
        boxes = targets_np[:, :4]
        if self.phase == 'train':
            labels = targets_np[:, 4:]
        else:
            # 若 test 阶段 labels 是一热，返回类别 id；否则默认最后一列是 id
            if targets_np.shape[1] > 5:
                labels = np.argmax(targets_np[:, 4:], axis=1).astype(np.int64)
            else:
                labels = targets_np[:, -1].astype(np.int64)

        if get_origin_image:
            return original_image, clip, boxes, labels
        else:
            return clip, boxes, labels


def _parse_img_size(img_cfg):
    """
    兼容 img_size 为 int / [H,W] / (H,W) / {'h':H,'w':W} / {'height':H,'width':W}
    返回 (H, W)
    """
    if isinstance(img_cfg, int):
        return (img_cfg, img_cfg)
    if isinstance(img_cfg, (list, tuple)):
        assert len(img_cfg) == 2, f"img_size 长度应为2，当前为 {img_cfg}"
        return (int(img_cfg[0]), int(img_cfg[1]))
    if isinstance(img_cfg, dict):
        if 'h' in img_cfg and 'w' in img_cfg:
            return (int(img_cfg['h']), int(img_cfg['w']))
        if 'height' in img_cfg and 'width' in img_cfg:
            return (int(img_cfg['height']), int(img_cfg['width']))
        raise ValueError(f"无法解析的 img_size 字段：{img_cfg}")
    raise TypeError(f"img_size 类型不支持：{type(img_cfg)}")


def build_ucf_dataset(config, phase):
    """
    期望 config 至少包含：
      - data_root
      - clip_length
      - sampling_rate
      - img_size  (int 或 [H,W] 或 (H,W) 或 dict)
      - num_classes
    """
    root_path = config['data_root']
    data_path = "rgb-images"
    ann_path = "labels"
    clip_length = int(config['clip_length'])
    sampling_rate = int(config['sampling_rate'])
    img_size = _parse_img_size(config.get('img_size', 224))
    num_classes = int(config['num_classes'])

    if phase == 'train':
        split_path = "trainlist.txt"
        transform = Augmentation(img_size=img_size)
    elif phase == 'test':
        split_path = "testlist.txt"
        transform = UCF_transform(img_size=img_size)
    else:
        raise ValueError("phase 必须是 'train' 或 'test'。")

    return UCF_dataset(root_path=root_path,
                       split_path=split_path,
                       data_path=data_path,
                       ann_path=ann_path,
                       clip_length=clip_length,
                       sampling_rate=sampling_rate,
                       img_size=img_size,
                       phase=phase,
                       num_classes=num_classes,
                       transform=transform)
