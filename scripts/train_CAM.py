# import os
# import sys
# import shutil
# import time
# import logging
# import random
# import glob
# import cv2
# import numpy as np
# import torch
# import torch.utils.data as data
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms.functional as FT
# import torch.optim as optim
# import tqdm
# from PIL import Image
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix
#
# from utils.gradflow_check import plot_grad_flow
# from utils.EMA import EMA
# from utils.build_config import build_config
# from cus_datasets.ucf.load_data import UCF_dataset
# from cus_datasets.collate_fn import collate_fn
# from cus_datasets.build_dataset import build_dataset
# from model.TSN.YOWOv3 import build_yowov3
# from utils.loss import build_loss
# from utils.warmup_lr import LinearWarmup
# from utils.flops import get_info
# from utils.box import non_max_suppression, box_iou
# from cus_datasets.ucf.transforms import UCF_transform
#
#
# # -------------------- 小工具 --------------------
# def _parse_img_size(img_size):
#     """返回 (H, W)"""
#     if isinstance(img_size, int):
#         return img_size, img_size
#     if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
#         return int(img_size[0]), int(img_size[1])
#     raise TypeError(f"img_size 格式不支持：{img_size!r}，请用 int 或 (H, W)")
#
#
# def _to_cuda_tensor(x, device, is_label=False, num_classes=None):
#     """
#     通用 numpy/list/tensor -> cuda tensor
#     is_label=True 时可选择转 one-hot
#     """
#     if isinstance(x, np.ndarray):
#         t = torch.from_numpy(x)
#     elif isinstance(x, torch.Tensor):
#         t = x
#     else:
#         t = torch.as_tensor(x)
#
#     if is_label:
#         # 明确是类别 id 的情况
#         if t.ndim == 1 or (t.ndim == 2 and t.shape[-1] == 1) or t.dtype in (torch.int32, torch.int64):
#             t = t.view(-1).long()
#             if num_classes is not None:
#                 t = F.one_hot(t, num_classes=num_classes).float()
#         else:
#             # 视作 one-hot
#             t = t.float()
#     else:
#         t = t.float()
#     return t.to(device, non_blocking=True)
#
#
# # -------------------- 训练主函数 --------------------
# def train_model(config):
#     # 保存配置文件
#     source_file = config['config_path']
#     destination_file = os.path.join(config['save_folder'], 'config.yaml')
#     os.makedirs(config['save_folder'], exist_ok=True)
#     shutil.copyfile(source_file, destination_file)
#
#     # 数据
#     dataset = build_dataset(config, phase='train')
#     dataloader = data.DataLoader(
#         dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=config['num_workers'],
#         pin_memory=True
#     )
#
#     # 模型
#     model = build_yowov3(config)
#     get_info(config, model)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.train()
#
#     criterion = build_loss(model, config)
#
#     # 优化器参数分组
#     g = [], [], []
#     bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
#     for v in model.modules():
#         for p_name, p in v.named_parameters(recurse=False):
#             if p_name == "bias":
#                 g[2].append(p)
#             elif p_name == "weight" and isinstance(v, bn):
#                 g[1].append(p)
#             else:
#                 g[0].append(p)
#
#     optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
#     optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
#     optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})
#
#     warmup_lr = LinearWarmup(config)
#     adjustlr_schedule = config['adjustlr_schedule']
#     acc_grad = config['acc_grad']
#     max_epoch = config['max_epoch']
#     lr_decay = config['lr_decay']
#     save_folder = config['save_folder']
#     num_classes = int(config['num_classes'])
#
#     torch.backends.cudnn.benchmark = True
#     cur_epoch = 1
#     loss_acc = 0.0
#     ema = EMA(model)
#
#     global best_map50, no_improvement_count
#     best_map50 = 0.0
#     no_improvement_count = 0
#
#     prev_save_path_ema = None
#     prev_save_path = None
#
#     # 训练循环
#     while (cur_epoch <= max_epoch) and (no_improvement_count < 100):
#         cnt_pram_update = 0
#         for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
#             batch_clip = batch_clip.to(device, non_blocking=True)
#             batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
#             batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=num_classes) for l in batch_labels]
#
#             outputs = model(batch_clip)
#
#             # 组 targets: [img_idx, x1,y1,x2,y2, one-hot...]
#             targets = []
#             for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
#                 nbox = bboxes.shape[0]
#                 if nbox == 0:
#                     continue
#                 if labels.ndim == 1:
#                     labels = F.one_hot(labels.long(), num_classes=num_classes).float()
#                 elif labels.ndim == 2 and labels.shape[1] != num_classes:
#                     labels = F.one_hot(labels.view(-1).long(), num_classes=num_classes).float()
#                 nclass = labels.shape[1]
#                 tgt = torch.zeros(nbox, 5 + nclass, device=device, dtype=torch.float32)
#                 tgt[:, 0] = i
#                 tgt[:, 1:5] = bboxes
#                 tgt[:, 5:] = labels
#                 targets.append(tgt)
#             if len(targets) == 0:
#                 optimizer.zero_grad(set_to_none=True)
#                 continue
#             targets = torch.cat(targets, dim=0)
#
#             loss = criterion(outputs, targets) / acc_grad
#             loss_acc += float(loss.item())
#             loss.backward()
#
#             if (iteration + 1) % acc_grad == 0:
#                 cnt_pram_update += 1
#                 if cur_epoch == 1:
#                     warmup_lr(optimizer, cnt_pram_update)
#                 nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)
#                 ema.update(model)
#
#                 print(f"epoch : {cur_epoch}, update : {cnt_pram_update}, loss = {loss_acc}", flush=True)
#                 with open(os.path.join(save_folder, "logging.txt"), "a") as f:
#                     f.write(f"epoch : {cur_epoch}, update : {cnt_pram_update}, loss = {loss_acc}\n")
#                 loss_acc = 0.0
#
#         # 学习率衰减
#         if cur_epoch in adjustlr_schedule:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= lr_decay
#
#         # 验证
#         eval_results = eval(config, model)
#
#         # 保存 best
#         if cur_epoch == 1:
#             save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
#             torch.save(ema.ema.state_dict(), save_path_ema)
#             prev_save_path_ema = save_path_ema
#
#             save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
#             torch.save(model.state_dict(), save_path)
#             prev_save_path = save_path
#
#             best_map50 = eval_results['map50']
#         elif eval_results['map50'] > best_map50:
#             best_map50 = eval_results['map50']
#             print(f"New best model found at epoch {cur_epoch} with map50: {best_map50}", flush=True)
#
#             save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
#             torch.save(ema.ema.state_dict(), save_path_ema)
#             save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
#             torch.save(model.state_dict(), save_path)
#
#             if prev_save_path_ema and os.path.isfile(prev_save_path_ema):
#                 os.remove(prev_save_path_ema)
#             if prev_save_path and os.path.isfile(prev_save_path):
#                 os.remove(prev_save_path)
#
#             prev_save_path_ema = save_path_ema
#             prev_save_path = save_path
#             no_improvement_count = 0
#         else:
#             no_improvement_count += 1
#
#         model.train()
#         print(f"Current epoch {cur_epoch} completed. Best map50: {best_map50}", flush=True)
#         cur_epoch += 1
#
#
# # -------------------- 验证并绘制混淆矩阵 --------------------
# @torch.no_grad()
# def eval(config, model):
#     dataset = build_dataset(config, phase='test')
#     dataloader = data.DataLoader(
#         dataset,
#         batch_size=16,          # 你当前设置的 batch_size / num_workers
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=16,
#         pin_memory=True
#     )
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval().to(device)
#
#     # 仅 mAP@0.5
#     iou_v = torch.tensor([0.5], device=device)
#     n_iou = iou_v.numel()
#
#     metrics = []
#     p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
#
#     total_detections = 0
#     total_ground_truths = 0
#
#     H, W = _parse_img_size(config['img_size'])
#
#     # 混淆矩阵数据收集
#     y_true_all, y_pred_all = [], []
#
#     for batch_clip, batch_bboxes, batch_labels in p_bar:
#         batch_clip = batch_clip.to(device, non_blocking=True)
#         batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
#         # 验证阶段，labels 以 id 形式
#         batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=None) for l in batch_labels]
#
#         # 组 targets: [img_idx, cls_id, x1,y1,x2,y2]
#         targets = []
#         for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
#             n = bboxes.shape[0]
#             if n == 0:
#                 continue
#
#             # labels: one-hot -> id；否则直接视作 id
#             if labels.ndim == 2 and labels.shape[1] > 1:
#                 cls_ids = labels.argmax(dim=1).long()
#             else:
#                 cls_ids = labels.view(-1).long()
#
#             tgt = torch.zeros(n, 6, device=device, dtype=torch.float32)
#             tgt[:, 0] = i
#             tgt[:, 1] = cls_ids.float()
#             tgt[:, 2:] = bboxes
#             targets.append(tgt)
#             total_ground_truths += n
#
#         if len(targets) == 0:
#             # 没有 GT 的 batch
#             outputs = model(batch_clip)
#             outputs = non_max_suppression(outputs, 0.5, 0.5)
#             continue
#
#         targets = torch.cat(targets, dim=0)
#
#         # 推理
#         outputs = model(batch_clip)
#
#         # 还原到像素坐标
#         targets[:, 2:] *= torch.tensor((W, H, W, H), device=device, dtype=torch.float32)
#
#         # NMS
#         outputs = non_max_suppression(outputs, 0.5, 0.5)
#
#         # 针对 batch 内每一张图
#         for i, output in enumerate(outputs):
#             labels = targets[targets[:, 0] == i, 1:]  # [cls, x1,y1,x2,y2]
#
#             if output is not None and output.shape[0] > 0:
#                 total_detections += output.shape[0]
#             else:
#                 # 无检测但有 GT 的情况，metrics 里只记空预测
#                 if labels.shape[0] > 0:
#                     correct = torch.zeros((0, n_iou), dtype=torch.bool, device=device)
#                     conf = torch.zeros(0, device=device)
#                     pred_cls = torch.zeros(0, device=device)
#                     target_cls = labels[:, 0].to(device)
#                     metrics.append((correct, conf, pred_cls, target_cls))
#
#                     # 混淆矩阵：认为未检测为类别 0（可自行修改策略）
#                     y_true_all.extend(labels[:, 0].detach().cpu().numpy().astype(int))
#                     y_pred_all.extend([0] * labels.shape[0])
#                 continue
#
#             # 有检测结果
#             detections = output.clone()
#             correct = torch.zeros(detections.shape[0], n_iou, dtype=torch.bool, device=device)
#
#             if labels.shape[0] > 0:
#                 tbox = labels[:, 1:5].clone()  # xyxy
#                 t_tensor = torch.cat((labels[:, 0:1], tbox), 1)  # [cls, x1,y1,x2,y2]
#                 iou = box_iou(t_tensor[:, 1:], detections[:, :4])  # [n_gt, n_det]
#                 correct_class = t_tensor[:, 0:1] == detections[:, 5]
#
#                 for j in range(n_iou):
#                     x = torch.where((iou >= iou_v[j]) & correct_class)
#                     if x[0].numel():
#                         matches = torch.cat(
#                             (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
#                         )  # [k, 3] -> [gt_idx, det_idx, iou]
#                         matches = matches.detach().cpu().numpy()
#                         if matches.shape[0] > 1:
#                             matches = matches[matches[:, 2].argsort()[::-1]]
#                             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#                         correct[matches[:, 1].astype(int), j] = True
#
#                 # ===== 混淆矩阵收集（简单一一配对策略） =====
#                 preds = detections[:, 5].detach().cpu().numpy().astype(int)
#                 gts = labels[:, 0].detach().cpu().numpy().astype(int)
#                 n_match = min(len(preds), len(gts))
#                 if n_match > 0:
#                     y_true_all.extend(gts[:n_match])
#                     y_pred_all.extend(preds[:n_match])
#
#             else:
#                 # 只有检测，没有 GT：对 AP 有意义，对混淆矩阵这里不加 y_true
#                 pass
#
#             metrics.append((
#                 correct,
#                 detections[:, 4],
#                 detections[:, 5],
#                 labels[:, 0] if labels.shape[0] > 0 else torch.tensor([], device=device)
#             ))
#
#     # -------------------- 汇总指标 --------------------
#     if len(metrics) > 0:
#         all_correct, all_conf, all_pred_cls, all_target_cls = [], [], [], []
#         for correct, conf, pred_cls, target_cls in metrics:
#             all_correct.append(correct.detach().cpu().numpy())
#             all_conf.append(conf.detach().cpu().numpy())
#             all_pred_cls.append(pred_cls.detach().cpu().numpy())
#             all_target_cls.append(target_cls.detach().cpu().numpy())
#
#         tp = np.concatenate(all_correct, axis=0) if len(all_correct) else np.zeros((0, n_iou), dtype=bool)
#         conf = np.concatenate(all_conf, axis=0) if len(all_conf) else np.zeros((0,), dtype=np.float32)
#         pred_cls = np.concatenate(all_pred_cls, axis=0) if len(all_pred_cls) else np.zeros((0,), dtype=np.float32)
#         target_cls = np.concatenate(all_target_cls, axis=0) if len(all_target_cls) else np.zeros((0,), dtype=np.float32)
#
#         if tp.shape[0] > 0 and target_cls.shape[0] > 0:
#             tp_count, fp_count, m_pre, m_rec, map50, mean_ap, ap_classes, ap_values = \
#                 compute_ap(tp, conf, pred_cls, target_cls)
#         else:
#             tp_count = fp_count = 0
#             m_pre = m_rec = map50 = mean_ap = 0.0
#             ap_classes, ap_values = [], []
#     else:
#         tp_count = fp_count = 0
#         m_pre = m_rec = map50 = mean_ap = 0.0
#         ap_classes, ap_values = [], []
#
#     # 打印
#     print("=" * 60)
#     print("VALIDATION RESULTS:")
#     print("=" * 60)
#     print(f"Total Ground Truth Objects: {total_ground_truths}")
#     print(f"Total Detections: {total_detections}")
#     print(f"True Positives: {tp_count}")
#     print(f"False Positives: {fp_count}")
#     print(f"Mean Precision: {m_pre:.6f}")
#     print(f"Mean Recall: {m_rec:.6f}")
#     print(f"mAP@0.5: {map50:.6f}")
#     print(f"mAP@0.5:0.95: {mean_ap:.6f}")
#     print(f"{'Class':<10}{'AP@0.5':>10}")
#     for cid, ap_val in zip(ap_classes, ap_values):
#         print(f"{int(cid):<10}{ap_val:.6f}")
#     print("=" * 60)
#
#     # 日志
#     log_path = os.path.join(config['save_folder'], "logging.txt")
#     with open(log_path, "a") as f:
#         f.write("=" * 60 + "\n")
#         f.write("VALIDATION RESULTS:\n")
#         f.write("=" * 60 + "\n")
#         f.write(f"Total Ground Truth Objects: {total_ground_truths}\n")
#         f.write(f"Total Detections: {total_detections}\n")
#         f.write(f"True Positives: {tp_count}\n")
#         f.write(f"False Positives: {fp_count}\n")
#         f.write(f"Mean Precision: {m_pre:.6f}\n")
#         f.write(f"Mean Recall: {m_rec:.6f}\n")
#         f.write(f"mAP@0.5: {map50:.6f}\n")
#         f.write(f"mAP@0.5:0.95: {mean_ap:.6f}\n")
#         f.write("Per-class AP@0.5:\n")
#         for cid, ap_val in zip(ap_classes, ap_values):
#             f.write(f"Class {int(cid):<5}: {ap_val:.6f}\n")
#         f.write("=" * 60 + "\n\n")
#
#     results = {
#         'total_gt': total_ground_truths,
#         'total_det': total_detections,
#         'tp': tp_count,
#         'fp': fp_count,
#         'm_pre': m_pre,
#         'm_rec': m_rec,
#         'map50': map50,
#         'mean_ap': mean_ap
#     }
#
#     # -------------------- 绘制混淆矩阵（蓝色） --------------------
#     if len(y_true_all) > 0 and len(y_pred_all) > 0:
#         # 与 yaml 对应的 4 类标签
#         idx2name = {0: 'creeping', 1: 'feeding', 2: 'flying', 3: 'mating'}
#         labels_txt = [idx2name[i] for i in range(4)]
#
#         cm = confusion_matrix(
#             y_true_all,
#             y_pred_all,
#             labels=[0, 1, 2, 3],
#             normalize='true'   # 归一化到 [0,1]，行和为 1
#         )
#
#         plt.figure(figsize=(4.5, 4))
#         sns.heatmap(
#             cm,
#             annot=True,
#             fmt=".3f",
#             cmap="Blues",       # 改为蓝色
#             xticklabels=labels_txt,
#             yticklabels=labels_txt,
#             cbar=True,
#             vmin=0,
#             vmax=1
#         )
#         plt.xlabel("Predict label")
#         plt.ylabel("True label")
#         plt.tight_layout()
#         save_path_cm = os.path.join(config['save_folder'], "confusion_matrix.png")
#         plt.savefig(save_path_cm, dpi=300)
#         plt.close()
#
#     model.float()
#     return results
#
#
# # -------------------- 计算 AP 的工具 --------------------
# def smooth(y, f=0.05):
#     nf = round(len(y) * f * 2) // 2 + 1
#     p = np.ones(nf // 2)
#     yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
#     return np.convolve(yp, np.ones(nf) / nf, mode='valid')
#
#
# def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
#     """
#     计算 AP，tp: [N, n_iou] 的布尔矩阵；
#     这里只用到 n_iou=1（即 IoU=0.5）的情况也兼容。
#     """
#     # 按置信度排序
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
#
#     unique_classes, nt = np.unique(target_cls, return_counts=True)
#     nc = unique_classes.shape[0]
#
#     p = np.zeros((nc, 1000))
#     r = np.zeros((nc, 1000))
#     ap = np.zeros((nc, tp.shape[1]))
#     px = np.linspace(0, 1, 1000)
#
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         nl = nt[ci]     # 该类 GT 数
#         no = i.sum()    # 该类预测数
#         if no == 0 or nl == 0:
#             continue
#
#         fpc = (1 - tp[i]).cumsum(0)
#         tpc = tp[i].cumsum(0)
#
#         recall = tpc / (nl + eps)
#         r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
#
#         precision = tpc / (tpc + fpc + eps)
#         p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
#
#         for j in range(tp.shape[1]):
#             m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
#             m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
#             m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
#             ap[ci, j] = np.trapz(m_pre, m_rec)
#
#     f1 = 2 * p * r / (p + r + eps)
#     idx = smooth(f1.mean(0), 0.1).argmax()
#     p, r, f1 = p[:, idx], r[:, idx], f1[:, idx]
#     tp_count = (r * nt).round()
#     fp_count = (tp_count / (p + eps) - tp_count).round()
#     ap50, apm = ap[:, 0], ap.mean(1)
#     m_pre, m_rec = p.mean(), r.mean()
#     map50, mean_ap = ap50.mean(), apm.mean()
#
#     return int(tp_count.sum()), int(fp_count.sum()), float(m_pre), float(m_rec), float(map50), float(mean_ap), unique_classes, ap50

import os
import sys
import shutil
import time
import logging
import random
import glob
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as FT
import torch.optim as optim
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from utils.gradflow_check import plot_grad_flow
from utils.EMA import EMA
from utils.build_config import build_config
from cus_datasets.ucf.load_data import UCF_dataset
from cus_datasets.collate_fn import collate_fn
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.loss import build_loss
from utils.warmup_lr import LinearWarmup
from utils.flops import get_info
from utils.box import non_max_suppression, box_iou
from cus_datasets.ucf.transforms import UCF_transform


# -------------------- 小工具 --------------------
def _parse_img_size(img_size):
    """返回 (H, W)"""
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise TypeError(f"img_size 格式不支持：{img_size!r}，请用 int 或 (H, W)")


def _to_cuda_tensor(x, device, is_label=False, num_classes=None):
    """
    通用 numpy/list/tensor -> cuda tensor
    is_label=True 时可选择转 one-hot
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)

    if is_label:
        if t.ndim == 1 or (t.ndim == 2 and t.shape[-1] == 1) or t.dtype in (torch.int32, torch.int64):
            t = t.view(-1).long()
            if num_classes is not None:
                t = F.one_hot(t, num_classes=num_classes).float()
        else:
            t = t.float()
    else:
        t = t.float()
    return t.to(device, non_blocking=True)


# ------------------- 自定义画框（加粗 & 字体黑色） --------------------
def draw_boxes_thick(image, boxes, clses, confs, mapping):
    """
    在 image 上画检测框（BGR）：
      - 加粗绿色框
      - 黑色文字 + 绿色底
    """
    box_color = (0, 255, 0)
    box_thickness = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_thickness = 2

    h, w = image.shape[:2]

    for box, cls_id, conf in zip(boxes, clses, confs):
        x1, y1, x2, y2 = box.tolist()
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w - 1, x2)))
        y2 = int(max(0, min(h - 1, y2)))

        cls_idx = int(cls_id.item()) if isinstance(cls_id, torch.Tensor) else int(cls_id)
        class_name = mapping.get(cls_idx, str(cls_idx))

        score = float(conf.item()) if isinstance(conf, torch.Tensor) else float(conf)
        label = f"{class_name}:{score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness=box_thickness)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x1
        text_y = y1 - 5
        if text_y - th - baseline < 0:
            text_y = y1 + th + baseline + 5

        cv2.rectangle(
            image,
            (text_x, text_y - th - baseline),
            (text_x + tw, text_y + baseline),
            box_color,
            thickness=-1
        )

        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )


# -------------------- 训练主函数 --------------------
def train_model(config):
    # 保存配置
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    os.makedirs(config['save_folder'], exist_ok=True)
    shutil.copyfile(source_file, destination_file)

    # 数据
    dataset = build_dataset(config, phase='train')
    dataloader = data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 模型
    model = build_yowov3(config)
    get_info(config, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = build_loss(model, config)

    # 优化器参数分组
    g = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=False):
            if p_name == "bias":
                g[2].append(p)
            elif p_name == "weight" and isinstance(v, bn):
                g[1].append(p)
            else:
                g[0].append(p)

    optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})

    warmup_lr = LinearWarmup(config)
    adjustlr_schedule = config['adjustlr_schedule']
    acc_grad = config['acc_grad']
    max_epoch = config['max_epoch']
    lr_decay = config['lr_decay']
    save_folder = config['save_folder']
    num_classes = int(config['num_classes'])

    torch.backends.cudnn.benchmark = True
    cur_epoch = 1
    loss_acc = 0.0
    ema = EMA(model)

    global best_map50, no_improvement_count
    best_map50 = 0.0
    no_improvement_count = 0

    prev_save_path_ema = None
    prev_save_path = None

    # 训练循环
    while (cur_epoch <= max_epoch) and (no_improvement_count < 100):
        cnt_pram_update = 0
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
            batch_clip = batch_clip.to(device, non_blocking=True)
            batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
            batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=num_classes) for l in batch_labels]

            outputs = model(batch_clip)

            # 组 targets: [img_idx, x1,y1,x2,y2, one-hot...]
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                if nbox == 0:
                    continue
                if labels.ndim == 1:
                    labels = F.one_hot(labels.long(), num_classes=num_classes).float()
                elif labels.ndim == 2 and labels.shape[1] != num_classes:
                    labels = F.one_hot(labels.view(-1).long(), num_classes=num_classes).float()
                nclass = labels.shape[1]
                tgt = torch.zeros(nbox, 5 + nclass, device=device, dtype=torch.float32)
                tgt[:, 0] = i
                tgt[:, 1:5] = bboxes
                tgt[:, 5:] = labels
                targets.append(tgt)
            if len(targets) == 0:
                optimizer.zero_grad(set_to_none=True)
                continue
            targets = torch.cat(targets, dim=0)

            loss = criterion(outputs, targets) / acc_grad
            loss_acc += float(loss.item())
            loss.backward()

            if (iteration + 1) % acc_grad == 0:
                cnt_pram_update += 1
                if cur_epoch == 1:
                    warmup_lr(optimizer, cnt_pram_update)
                nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

                print(f"epoch : {cur_epoch}, update : {cnt_pram_update}, loss = {loss_acc}", flush=True)
                with open(os.path.join(save_folder, "logging.txt"), "a") as f:
                    f.write(f"epoch : {cur_epoch}, update : {cnt_pram_update}, loss = {loss_acc}\n")
                loss_acc = 0.0

        # 学习率衰减
        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

        # 验证
        eval_results = eval(config, model)

        # 保存 best
        if cur_epoch == 1:
            save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
            torch.save(ema.ema.state_dict(), save_path_ema)
            prev_save_path_ema = save_path_ema

            save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
            torch.save(model.state_dict(), save_path)
            prev_save_path = save_path

            best_map50 = eval_results['map50']
        elif eval_results['map50'] > best_map50:
            best_map50 = eval_results['map50']
            print(f"New best model found at epoch {cur_epoch} with map50: {best_map50}", flush=True)

            save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
            torch.save(ema.ema.state_dict(), save_path_ema)
            save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
            torch.save(model.state_dict(), save_path)

            if prev_save_path_ema and os.path.isfile(prev_save_path_ema):
                os.remove(prev_save_path_ema)
            if prev_save_path and os.path.isfile(prev_save_path):
                os.remove(prev_save_path)

            prev_save_path_ema = save_path_ema
            prev_save_path = save_path
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        model.train()
        print(f"Current epoch {cur_epoch} completed. Best map50: {best_map50}", flush=True)
        cur_epoch += 1


# -------------------- 验证并绘制混淆矩阵 + 保存检测图片 --------------------
@torch.no_grad()
def eval(config, model):
    dataset = build_dataset(config, phase='test')
    dataloader = data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=16,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    iou_v = torch.tensor([0.5], device=device)
    n_iou = iou_v.numel()

    metrics = []
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    total_detections = 0
    total_ground_truths = 0

    H, W = _parse_img_size(config['img_size'])

    y_true_all, y_pred_all = [], []

    # 4 类标签
    idx2name = {0: 'creeping', 1: 'feeding', 2: 'flying', 3: 'mating'}

    # 保存检测图
    save_vis_folder = os.path.join(config['save_folder'], "detections")
    os.makedirs(save_vis_folder, exist_ok=True)
    save_img_idx = 0

    for batch_clip, batch_bboxes, batch_labels in p_bar:
        batch_clip = batch_clip.to(device, non_blocking=True)
        batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
        batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=None) for l in batch_labels]

        # 组 targets: [img_idx, cls_id, x1,y1,x2,y2]
        targets = []
        for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            n = bboxes.shape[0]
            if n == 0:
                continue

            if labels.ndim == 2 and labels.shape[1] > 1:
                cls_ids = labels.argmax(dim=1).long()
            else:
                cls_ids = labels.view(-1).long()

            tgt = torch.zeros(n, 6, device=device, dtype=torch.float32)
            tgt[:, 0] = i
            tgt[:, 1] = cls_ids.float()
            tgt[:, 2:] = bboxes
            targets.append(tgt)
            total_ground_truths += n

        targets = torch.cat(targets, dim=0) if len(targets) > 0 else None

        # 推理
        outputs = model(batch_clip)

        # 还原到像素坐标
        if targets is not None:
            targets[:, 2:] *= torch.tensor((W, H, W, H), device=device, dtype=torch.float32)

        # NMS
        outputs = non_max_suppression(outputs, 0.5, 0.5)

        # 每一张图
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:] if targets is not None else torch.zeros((0, 5), device=device)

            # --------- 从 batch_clip 恢复一帧为 3 通道图，并正确反归一化 ---------
            raw = batch_clip[i].detach().cpu()  # 可能是 [T,C,H,W] 或 [C,H,W]

            if raw.ndim == 4:          # [T,C,H,W]
                frame = raw[0]         # 第 1 帧 [C,H,W]
            elif raw.ndim == 3:        # [C,H,W]
                frame = raw
            else:
                raise ValueError(f"Unsupported clip shape: {raw.shape}")

            if frame.ndim != 3:
                raise ValueError(f"Frame ndim must be 3, got {frame.ndim}, shape={frame.shape}")

            frame_np = frame.numpy()
            C = frame_np.shape[0]

            # 通道处理：确保得到一个 3×H×W 的“可视化图像”
            if C == 3:
                rgb = frame_np
            elif C == 1:
                # 单通道 → 复制为 3 通道
                rgb = np.repeat(frame_np, 3, axis=0)
            elif C > 3:
                # 多通道（如 clip12 的 12 通道），只取第 0 个通道作为灰度，再复制为 3 通道
                single = frame_np[0]                      # [H,W]
                rgb = np.stack([single, single, single], axis=0)  # [3,H,W]
            else:
                raise ValueError(f"Unsupported channel number C={C} for visualization")

            # 数值范围判断：若最大值 <= 1.5，则认为是 [0,1] 归一化，需要乘 255；
            # 否则认为已经是 0~255，不再放大，避免“炸图”。
            vmin, vmax = float(rgb.min()), float(rgb.max())
            if vmax <= 1.5:
                rgb = rgb * 255.0

            rgb = np.clip(rgb, 0, 255).astype(np.uint8)      # [3,H,W]
            rgb = np.transpose(rgb, (1, 2, 0))               # [H,W,3]
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # 仅做 RGB→BGR 通道互换
            # -------------------------------------------------------------

            if output is not None and output.shape[0] > 0:
                boxes = output[:, :4].detach().cpu()
                confs = output[:, 4].detach().cpu()
                clses = output[:, 5].detach().cpu()

                draw_boxes_thick(frame_bgr, boxes, clses, confs, idx2name)

            save_path_img = os.path.join(save_vis_folder, f"img_{save_img_idx:05d}.jpg")
            cv2.imwrite(save_path_img, frame_bgr)
            save_img_idx += 1

            if output is not None:
                total_detections += output.shape[0]

            # ------ AP 统计 ------
            if labels.shape[0] == 0:
                if output is not None and output.shape[0] > 0:
                    correct = torch.zeros((output.shape[0], n_iou), dtype=torch.bool, device=device)
                    metrics.append((correct, output[:, 4], output[:, 5], torch.zeros(0, device=device)))
                else:
                    correct = torch.zeros((0, n_iou), dtype=torch.bool, device=device)
                    metrics.append((correct,
                                    torch.zeros(0, device=device),
                                    torch.zeros(0, device=device),
                                    torch.zeros(0, device=device)))
                continue

            detections = output.clone() if output is not None else torch.zeros((0, 6), device=device)
            correct = torch.zeros(detections.shape[0], n_iou, dtype=torch.bool, device=device)

            if detections.shape[0] > 0:
                tbox = labels[:, 1:5].clone()
                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)  # [cls, x1,y1,x2,y2]
                iou = box_iou(t_tensor[:, 1:], detections[:, :4])  # [n_gt, n_det]
                correct_class = t_tensor[:, 0:1] == detections[:, 5]

                for j in range(n_iou):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].numel():
                        matches = torch.cat(
                            (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1
                        )
                        matches = matches.detach().cpu().numpy()
                        if matches.shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True

                preds = detections[:, 5].detach().cpu().numpy().astype(int)
                gts = labels[:, 0].detach().cpu().numpy().astype(int)
                n_match = min(len(preds), len(gts))
                if n_match > 0:
                    y_true_all.extend(gts[:n_match])
                    y_pred_all.extend(preds[:n_match])

            metrics.append((
                correct,
                detections[:, 4] if detections.shape[0] > 0 else torch.zeros(0, device=device),
                detections[:, 5] if detections.shape[0] > 0 else torch.zeros(0, device=device),
                labels[:, 0]
            ))

    # -------------------- 汇总指标 --------------------
    if len(metrics) > 0:
        all_correct, all_conf, all_pred_cls, all_target_cls = [], [], [], []
        for correct, conf, pred_cls, target_cls in metrics:
            all_correct.append(correct.detach().cpu().numpy())
            all_conf.append(conf.detach().cpu().numpy())
            all_pred_cls.append(pred_cls.detach().cpu().numpy())
            all_target_cls.append(target_cls.detach().cpu().numpy())

        tp = np.concatenate(all_correct, axis=0) if len(all_correct) else np.zeros((0, n_iou), dtype=bool)
        conf = np.concatenate(all_conf, axis=0) if len(all_conf) else np.zeros((0,), dtype=np.float32)
        pred_cls = np.concatenate(all_pred_cls, axis=0) if len(all_pred_cls) else np.zeros((0,), dtype=np.float32)
        target_cls = np.concatenate(all_target_cls, axis=0) if len(all_target_cls) else np.zeros((0,), dtype=np.float32)

        if tp.shape[0] > 0 and target_cls.shape[0] > 0:
            tp_count, fp_count, m_pre, m_rec, map50, mean_ap, ap_classes, ap_values = \
                compute_ap(tp, conf, pred_cls, target_cls)
        else:
            tp_count = fp_count = 0
            m_pre = m_rec = map50 = mean_ap = 0.0
            ap_classes, ap_values = [], []
    else:
        tp_count = fp_count = 0
        m_pre = m_rec = map50 = mean_ap = 0.0
        ap_classes, ap_values = [], []

    print("=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)
    print(f"Total Ground Truth Objects: {total_ground_truths}")
    print(f"Total Detections: {total_detections}")
    print(f"True Positives: {tp_count}")
    print(f"False Positives: {fp_count}")
    print(f"Mean Precision: {m_pre:.6f}")
    print(f"Mean Recall: {m_rec:.6f}")
    print(f"mAP@0.5: {map50:.6f}")
    print(f"mAP@0.5:0.95: {mean_ap:.6f}")
    print(f"{'Class':<10}{'AP@0.5':>10}")
    for cid, ap_val in zip(ap_classes, ap_values):
        print(f"{int(cid):<10}{ap_val:.6f}")
    print("=" * 60)

    log_path = os.path.join(config['save_folder'], "logging.txt")
    with open(log_path, "a") as f:
        f.write("=" * 60 + "\n")
        f.write("VALIDATION RESULTS:\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Ground Truth Objects: {total_ground_truths}\n")
        f.write(f"Total Detections: {total_detections}\n")
        f.write(f"True Positives: {tp_count}\n")
        f.write(f"False Positives: {fp_count}\n")
        f.write(f"Mean Precision: {m_pre:.6f}\n")
        f.write(f"Mean Recall: {m_rec:.6f}\n")
        f.write(f"mAP@0.5: {map50:.6f}\n")
        f.write(f"mAP@0.5:0.95: {mean_ap:.6f}\n")
        f.write("Per-class AP@0.5:\n")
        for cid, ap_val in zip(ap_classes, ap_values):
            f.write(f"Class {int(cid):<5}: {ap_val:.6f}\n")
        f.write("=" * 60 + "\n\n")

    results = {
        'total_gt': total_ground_truths,
        'total_det': total_detections,
        'tp': tp_count,
        'fp': fp_count,
        'm_pre': m_pre,
        'm_rec': m_rec,
        'map50': map50,
        'mean_ap': mean_ap
    }

    # -------------------- 混淆矩阵 --------------------
    if len(y_true_all) > 0 and len(y_pred_all) > 0:
        labels_txt = [idx2name[i] for i in range(4)]

        cm = confusion_matrix(
            y_true_all,
            y_pred_all,
            labels=[0, 1, 2, 3],
            normalize='true'
        )

        plt.figure(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            xticklabels=labels_txt,
            yticklabels=labels_txt,
            cbar=True,
            vmin=0,
            vmax=1
        )
        plt.xlabel("Predict label")
        plt.ylabel("True label")
        plt.tight_layout()
        save_path_cm = os.path.join(config['save_folder'], "confusion_matrix.png")
        plt.savefig(save_path_cm, dpi=300)
        plt.close()

    model.float()
    return results


# -------------------- 计算 AP 的工具 --------------------
def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')


def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    计算 AP，tp: [N, n_iou] 布尔矩阵；
    """
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]

    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))
    ap = np.zeros((nc, tp.shape[1]))
    px = np.linspace(0, 1, 1000)

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        nl = nt[ci]
        no = i.sum()
        if no == 0 or nl == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        recall = tpc / (nl + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        precision = tpc / (tpc + fpc + eps)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        for j in range(tp.shape[1]):
            m_rec = np.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = np.concatenate(([1.0], precision[:, j], [0.0]))
            m_pre = np.flip(np.maximum.accumulate(np.flip(m_pre)))
            ap[ci, j] = np.trapz(m_pre, m_rec)

    f1 = 2 * p * r / (p + r + eps)
    idx = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, idx], r[:, idx], f1[:, idx]
    tp_count = (r * nt).round()
    fp_count = (tp_count / (p + eps) - tp_count).round()
    ap50, apm = ap[:, 0], ap.mean(1)
    m_pre, m_rec = p.mean(), r.mean()
    map50, mean_ap = ap50.mean(), apm.mean()

    return int(tp_count.sum()), int(fp_count.sum()), float(m_pre), float(m_rec), float(map50), float(mean_ap), unique_classes, ap50
