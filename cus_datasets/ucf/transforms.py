# from . import ucf_config
# import torch
# import numpy as np
#
#
# class UCF_transform():
#     """
#     Args:
#         clip  : list of (num_frame) np.array [H, W, C] (BGR order, 0..1)
#         boxes : list of (num_frame) list of (num_box, in ucf101-24 = 1) np.array [(x, y, w, h)] relative coordinate
#
#     Return:
#         clip  : torch.tensor [C, num_frame, H, W] (RGB order, 0..1)
#         boxes : not change
#     """
#
#     def __init__(self, img_size):
#         self.img_size = img_size
#         pass
#
#     def to_tensor(self, video_clip):
#         return [F.to_tensor(image) for image in video_clip]
#
#     def normalize(self, clip, mean=ucf_config.MEAN, std=ucf_config.STD):
#         mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1, 1)
#         std = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1, 1)
#         clip -= mean
#         clip /= std
#         return clip
#
#     def __call__(self, clip, targets):
#         W, H = clip[-1].size
#         targets[:, :4] /= np.array([W, H, W, H])
#         clip = [img.resize([self.img_size, self.img_size]) for img in clip]
#         clip = self.to_tensor(clip)
#         clip = torch.stack(clip, dim=1)
#         clip = self.normalize(clip)
#         targets = torch.as_tensor(targets).float()
#         return clip, targets
#
#
# import torch
# import torch.utils.data as data
# import argparse
# import yaml
# import os
# import cv2
# import pickle
# import numpy as np
# from PIL import Image
# import sys
#
# import random
# import torchvision.transforms.functional as F
#
#
# class Augmentation(object):
#     def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
#         self.img_size = img_size
#         self.jitter = jitter
#         self.hue = hue
#         self.saturation = saturation
#         self.exposure = exposure
#
#     def rand_scale(self, s):
#         scale = random.uniform(1, s)
#
#         if random.randint(0, 1):
#             return scale
#
#         return 1. / scale
#
#     def random_distort_image(self, video_clip):
#         dhue = random.uniform(-self.hue, self.hue)
#         dsat = self.rand_scale(self.saturation)
#         dexp = self.rand_scale(self.exposure)
#
#         video_clip_ = []
#         for image in video_clip:
#             image = image.convert('HSV')
#             cs = list(image.split())
#             cs[1] = cs[1].point(lambda i: i * dsat)
#             cs[2] = cs[2].point(lambda i: i * dexp)
#
#             def change_hue(x):
#                 x += dhue * 255
#                 if x > 255:
#                     x -= 255
#                 if x < 0:
#                     x += 255
#                 return x
#
#             cs[0] = cs[0].point(change_hue)
#             image = Image.merge(image.mode, tuple(cs))
#
#             image = image.convert('RGB')
#
#             video_clip_.append(image)
#
#         return video_clip_
#
#     def random_crop(self, video_clip, width, height):
#         dw = int(width * self.jitter)
#         dh = int(height * self.jitter)
#
#         pleft = random.randint(-dw, dw)
#         pright = random.randint(-dw, dw)
#         ptop = random.randint(-dh, dh)
#         pbot = random.randint(-dh, dh)
#
#         swidth = width - pleft - pright
#         sheight = height - ptop - pbot
#
#         sx = float(swidth) / width
#         sy = float(sheight) / height
#
#         dx = (float(pleft) / width) / sx
#         dy = (float(ptop) / height) / sy
#
#         # random crop
#         cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]
#
#         return cropped_clip, dx, dy, sx, sy
#
#     def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
#         sx, sy = 1. / sx, 1. / sy
#         # apply deltas on bbox
#         target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx))
#         target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy))
#         target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx))
#         target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy))
#
#         # refine target
#         refine_target = []
#         for i in range(target.shape[0]):
#             tgt = target[i]
#             bw = (tgt[2] - tgt[0]) * ow
#             bh = (tgt[3] - tgt[1]) * oh
#
#             if bw < 1. or bh < 1.:
#                 continue
#
#             refine_target.append(tgt)
#
#         refine_target = np.array(refine_target).reshape(-1, target.shape[-1])
#
#         return refine_target
#
#     def to_tensor(self, video_clip):
#         return [F.to_tensor(image) for image in video_clip]
#
#     def normalization(self, video_clip):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
#
#         video_clip = (video_clip - mean) / std
#         return video_clip
#
#     def __call__(self, video_clip, target):
#         # Initialize Random Variables
#         oh = video_clip[-1].height
#         ow = video_clip[-1].width
#
#         # random crop
#         video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)
#
#         # resize
#         video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]
#
#         # random flip
#         flip = random.randint(0, 1)
#         if flip:
#             video_clip = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in video_clip]
#
#         # distort
#         video_clip = self.random_distort_image(video_clip)
#
#         # process target
#         if target is not None:
#             target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
#             if flip:
#                 target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
#         else:
#             target = np.array([])
#
#         # to tensor
#         video_clip = self.to_tensor(video_clip)
#         video_clip = torch.stack(video_clip, dim=1)
#
#         video_clip = self.normalization(video_clip)
#
#         target = torch.as_tensor(target).float()
#
#         return video_clip, target
#
#
# def UCF_collate_fn(batch_data):
#     clips = []
#     boxes = []
#     labels = []
#     for b in batch_data:
#         clips.append(b[0])
#         boxes.append(b[1])
#         labels.append(b[2])
#
#     clips = torch.stack(clips, dim=0)  # [batch_size, num_frame, C, H, W]
#     return clips, boxes, labels
#
import random
from typing import Tuple, Union, List, Dict

import numpy as np
from PIL import Image

import torch
import torchvision.transforms.functional as F


# ---------------------------------------
# 将 img_size 解析为:
#   size_hw = (H, W)  —— 便于 boxes 计算
#   size_wh = (W, H)  —— 传给 PIL.Image.resize
# ---------------------------------------
def _parse_img_size(img_size: Union[int, Tuple[int, int], List[int], Dict]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
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


class UCF_transform:
    """
    Args:
        clip: List[PIL.Image], len = T
        targets: numpy array (N, 4+K)，前4列为 (x1,y1,x2,y2) 像素坐标
    Return:
        clip: torch.Tensor [C, T, H, W]，已标准化
        targets: torch.FloatTensor (N, 4+K)；前4列为**相对坐标**
    """

    def __init__(self, img_size: Union[int, Tuple[int, int], List[int], Dict]):
        self.size_hw, self.size_wh = _parse_img_size(img_size)

    @staticmethod
    def _to_tensor(video_clip: List[Image.Image]):
        return [F.to_tensor(image) for image in video_clip]

    @staticmethod
    def _normalize(clip: torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (clip - mean) / std

    def __call__(self, clip: List[Image.Image], targets: np.ndarray):
        # 像素 -> 相对坐标
        W, H = clip[-1].size  # PIL: (W, H)
        if isinstance(targets, np.ndarray) and targets.size > 0:
            denom = np.array([W, H, W, H], dtype=np.float32)
            targets[:, :4] = targets[:, :4].astype(np.float32) / denom

        # resize 到 (W,H) = self.size_wh
        clip = [img.resize(self.size_wh, resample=Image.BILINEAR) for img in clip]

        # to tensor & normalize
        clip = self._to_tensor(clip)
        clip = torch.stack(clip, dim=1)  # [C, T, H, W]
        clip = self._normalize(clip)

        targets = torch.as_tensor(targets).float() if isinstance(targets, np.ndarray) else torch.zeros((0, 5), dtype=torch.float32)
        return clip, targets


class Augmentation:
    """
    训练增强：
    - 随机裁剪 -> resize -> 随机翻转 -> 颜色扰动 -> 标准化
    输入 target 前4列视为像素坐标；内部会转为相对坐标并应用仿射
    """

    def __init__(self,
                 img_size: Union[int, Tuple[int, int], List[int], Dict] = 224,
                 jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.size_hw, self.size_wh = _parse_img_size(img_size)
        self.jitter = float(jitter)
        self.hue = float(hue)
        self.saturation = float(saturation)
        self.exposure = float(exposure)

    @staticmethod
    def _rand_scale(s):
        scale = random.uniform(1.0, s)
        return scale if random.randint(0, 1) else 1.0 / scale

    def _random_distort(self, video_clip: List[Image.Image]):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        out = []
        for image in video_clip:
            image = image.convert('HSV')
            h, s, v = image.split()
            s = s.point(lambda i: i * dsat)
            v = v.point(lambda i: i * dexp)

            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            h = h.point(change_hue)
            image = Image.merge('HSV', (h, s, v)).convert('RGB')
            out.append(image)
        return out

    def _random_crop(self, video_clip: List[Image.Image], width: int, height: int):
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        swidth = max(2, width - pleft - pright)
        sheight = max(2, height - ptop - pbot)

        sx = float(swidth) / width
        sy = float(sheight) / height

        dx = (float(pleft) / width) / sx
        dy = (float(ptop) / height) / sy

        cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]
        return cropped_clip, dx, dy, sx, sy

    @staticmethod
    def _apply_bbox(target: np.ndarray, ow: int, oh: int, dx: float, dy: float, sx: float, sy: float) -> np.ndarray:
        if not isinstance(target, np.ndarray) or target.size == 0:
            return np.zeros((0, 0), dtype=np.float32)

        target = target.astype(np.float32, copy=True)
        # 注意原实现里用的是倒数
        sx, sy = 1.0 / sx, 1.0 / sy

        # 像素 -> 相对，并应用随机裁剪仿射
        target[..., 0] = np.minimum(0.999, np.maximum(0.0, target[..., 0] / ow * sx - dx))
        target[..., 1] = np.minimum(0.999, np.maximum(0.0, target[..., 1] / oh * sy - dy))
        target[..., 2] = np.minimum(0.999, np.maximum(0.0, target[..., 2] / ow * sx - dx))
        target[..., 3] = np.minimum(0.999, np.maximum(0.0, target[..., 3] / oh * sy - dy))

        # 过滤过小框
        refine = []
        D = target.shape[-1]
        for i in range(target.shape[0]):
            x1, y1, x2, y2 = target[i, :4]
            bw = (x2 - x1) * ow
            bh = (y2 - y1) * oh
            if bw >= 1.0 and bh >= 1.0:
                refine.append(target[i])

        if len(refine) == 0:
            return np.zeros((0, D), dtype=target.dtype)
        return np.asarray(refine, dtype=target.dtype).reshape(-1, D)

    @staticmethod
    def _to_tensor(video_clip: List[Image.Image]):
        return [F.to_tensor(image) for image in video_clip]

    @staticmethod
    def _normalize(video_clip: torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        return (video_clip - mean) / std

    def __call__(self, video_clip: List[Image.Image], target: np.ndarray):
        # 原始尺寸
        oh = video_clip[-1].height
        ow = video_clip[-1].width

        # 1) 随机裁剪
        video_clip, dx, dy, sx, sy = self._random_crop(video_clip, ow, oh)

        # 2) resize
        video_clip = [img.resize(self.size_wh, resample=Image.BILINEAR) for img in video_clip]

        # 3) 随机水平翻转
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for img in video_clip]

        # 4) 颜色扰动
        video_clip = self._random_distort(video_clip)

        # 5) 处理目标
        if isinstance(target, np.ndarray) and target.size > 0:
            target = self._apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if target.size > 0 and flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
        else:
            # 若无法确定列数则假定 5 列（4坐标 + 1标签）
            ncol = int(target.shape[1]) if isinstance(target, np.ndarray) and target.ndim == 2 else 5
            target = np.zeros((0, ncol), dtype=np.float32)

        # 6) to tensor & normalize
        video_clip = self._to_tensor(video_clip)
        video_clip = torch.stack(video_clip, dim=1)  # [C, T, H, W]
        video_clip = self._normalize(video_clip)

        target = torch.as_tensor(target).float()
        return video_clip, target


def UCF_collate_fn(batch_data):
    """
    返回:
        clips:  torch.Tensor [B, C, T, H, W]
        boxes:  List[torch.FloatTensor]，每个元素形状 (Ni, 4)
        labels: List[torch.Tensor]，训练时为 Float(one-hot)，测试时为 Long(类别id)
    这样 train.py 里对 boxes/labels 的 .to("cuda") 就不会再报错。
    """
    clips = []
    boxes_list = []
    labels_list = []

    for sample in batch_data:
        clip, boxes, labels = sample  # clip: Tensor [C,T,H,W]; boxes/labels: numpy 或 Tensor

        # clip 直接收集（已是 Tensor）
        clips.append(clip)

        # --- boxes -> torch.float32 ---
        if isinstance(boxes, np.ndarray):
            boxes_t = torch.from_numpy(boxes).float()
        elif isinstance(boxes, torch.Tensor):
            boxes_t = boxes.float()
        else:
            # list 等其它情况
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        boxes_list.append(boxes_t)

        # --- labels -> 训练: float(one-hot)；测试: long(ids) ---
        if isinstance(labels, np.ndarray):
            if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1):
                labels_t = torch.from_numpy(labels.reshape(-1)).long()
            else:
                labels_t = torch.from_numpy(labels).float()
        elif isinstance(labels, torch.Tensor):
            if labels.ndim == 1 or (labels.ndim == 2 and labels.shape[1] == 1) or labels.dtype in (torch.int32, torch.int64):
                labels_t = labels.reshape(-1).long()
            else:
                labels_t = labels.float()
        else:
            # 其它类型（如 list[int] / list[list[float]]）
            arr = np.asarray(labels)
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
                labels_t = torch.from_numpy(arr.reshape(-1)).long()
            else:
                labels_t = torch.from_numpy(arr).float()
        labels_list.append(labels_t)

    clips = torch.stack(clips, dim=0)  # [B, C, T, H, W]
    return clips, boxes_list, labels_list
