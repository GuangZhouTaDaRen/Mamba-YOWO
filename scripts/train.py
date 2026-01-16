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
# import numpy
#
# from math import sqrt
# from utils.gradflow_check import plot_grad_flow
# from utils.EMA import EMA
# import logging
# from utils.build_config import build_config
# from cus_datasets.ucf.load_data import UCF_dataset
# from cus_datasets.collate_fn import collate_fn
# from cus_datasets.build_dataset import build_dataset
# from model.TSN.YOWOv3 import build_yowov3
# from utils.loss import build_loss
# from utils.warmup_lr import LinearWarmup
# import shutil
# from utils.flops import get_info
# from utils.box import non_max_suppression, box_iou
# from evaluator.eval import compute_ap
# import tqdm
# from cus_datasets.ucf.transforms import UCF_transform
#
#
# def train_model(config):
#     # Save config file
#     #######################################################
#     source_file = config['config_path']
#     destination_file = os.path.join(config['save_folder'], 'config.yaml')
#     shutil.copyfile(source_file, destination_file)
#     #######################################################
#
#     # create dataloader, model, criterion
#     ####################################################
#     dataset = build_dataset(config, phase='train')
#
#     dataloader = data.DataLoader(dataset, config['batch_size'], True, collate_fn=collate_fn
#                                  , num_workers=config['num_workers'], pin_memory=True)
#
#     model = build_yowov3(config)
#     get_info(config, model)
#     model.to("cuda")
#     model.train()
#
#     criterion = build_loss(model, config)
#     #####################################################
#
#     g = [], [], []  # optimizer parameter groups
#     bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
#     for v in model.modules():
#         for p_name, p in v.named_parameters(recurse=0):
#             if p_name == "bias":  # bias (no decay)
#                 g[2].append(p)
#             elif p_name == "weight" and isinstance(v, bn):  # weight (no decay)
#                 g[1].append(p)
#             else:
#                 g[0].append(p)  # weight (with decay)
#
#     optimizer = torch.optim.AdamW(g[0], lr=config['lr'], weight_decay=config['weight_decay'])
#     optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
#     optimizer.add_param_group({"params": g[2], "weight_decay": 0.0})
#
#     warmup_lr = LinearWarmup(config)
#
#     adjustlr_schedule = config['adjustlr_schedule']
#     acc_grad = config['acc_grad']
#     max_epoch = config['max_epoch']
#     lr_decay = config['lr_decay']
#     save_folder = config['save_folder']
#
#     torch.backends.cudnn.benchmark = True
#     cur_epoch = 1
#     loss_acc = 0.0
#     ema = EMA(model)
#
#     # 增加best_map50，后续用于判断保存map50最高的epoch权重
#     global best_map50
#     global no_improvement_count
#     best_map50 = 0.0
#     no_improvement_count = 0
#
#     # 用于保存上一个epoch的模型路径
#     prev_save_path_ema = None
#     prev_save_path = None
#
#     # 增加 no_improvement_count计数器判断，模型在100轮中，map50未有增高，则自动停止训练，防止过拟合
#     while (cur_epoch <= max_epoch) and (no_improvement_count < 100):
#         cnt_pram_update = 0
#         for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
#
#             batch_size = batch_clip.shape[0]
#             batch_clip = batch_clip.to("cuda")
#             for idx in range(batch_size):
#                 batch_bboxes[idx] = batch_bboxes[idx].to("cuda")
#                 batch_labels[idx] = batch_labels[idx].to("cuda")
#
#             outputs = model(batch_clip)
#
#             targets = []
#             for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
#                 nbox = bboxes.shape[0]
#                 nclass = labels.shape[1]
#                 target = torch.Tensor(nbox, 5 + nclass)
#                 target[:, 0] = i
#                 target[:, 1:5] = bboxes
#                 target[:, 5:] = labels
#                 targets.append(target)
#
#             targets = torch.cat(targets, dim=0)
#
#             loss = criterion(outputs, targets) / acc_grad
#             loss_acc += loss.item()
#             loss.backward()
#
#             if (iteration + 1) % acc_grad == 0:
#                 cnt_pram_update = cnt_pram_update + 1
#                 if cur_epoch == 1:
#                     warmup_lr(optimizer, cnt_pram_update)
#                 nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 ema.update(model)
#
#                 print("epoch : {}, update : {}, loss = {}".format(cur_epoch, cnt_pram_update, loss_acc), flush=True)
#                 # 更改logging.txt为追加模式（"a"），可记录所有epoch loss信息
#                 with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:
#                     # 添加换行符，便于阅读
#                     f.write("epoch : {}, update : {}, loss = {}\n".format(cur_epoch, cnt_pram_update, loss_acc))
#
#                 loss_acc = 0.0
#
#         if cur_epoch in adjustlr_schedule:
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] *= lr_decay
#
#         # 进行验证（eval函数会将模型设置为验证模式）
#         eval_results = eval(config, model)
#
#         # 判断是否为当前最佳模型并保存
#         if cur_epoch == 1:
#             # 保存第一个epoch的模型
#             save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
#             torch.save(ema.ema.state_dict(), save_path_ema)
#             prev_save_path_ema = save_path_ema
#
#             save_path = os.path.join(save_folder, "epoch_" + str(cur_epoch) + ".pth")
#             torch.save(model.state_dict(), save_path)
#
#             prev_save_path = save_path
#             best_map50 = eval_results['map50']
#
#             # 记录详细验证结果到日志
#             with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:
#                 f.write("=" * 50 + "\n")
#                 f.write("EPOCH {} VALIDATION RESULTS:\n".format(cur_epoch))
#                 f.write("=" * 50 + "\n")
#                 f.write("True Positives: {}\n".format(eval_results['tp']))
#                 f.write("False Positives: {}\n".format(eval_results['fp']))
#                 f.write("Mean Precision: {:.6f}\n".format(eval_results['m_pre']))
#                 f.write("Mean Recall: {:.6f}\n".format(eval_results['m_rec']))
#                 f.write("mAP@0.5: {:.6f}\n".format(eval_results['map50']))
#                 f.write("mAP@0.5:0.95: {:.6f}\n".format(eval_results['mean_ap']))
#                 f.write("Best mAP@0.5 so far: {:.6f}\n".format(best_map50))
#                 f.write("=" * 50 + "\n\n")
#
#         elif eval_results['map50'] > best_map50:
#             best_map50 = eval_results['map50']
#             print("New best model found at epoch {} with map50: {}".format(cur_epoch, best_map50), flush=True)
#
#             # 保存当前最佳模型(一个ema后model，一个原始model)
#             save_path_ema = os.path.join(save_folder, "ema_epoch_" + str(cur_epoch) + ".pth")
#             torch.save(ema.ema.state_dict(), save_path_ema)
#
#             save_path = os.path.join(save_folder, "epoch_" + str(cur_epoch) + ".pth")
#             torch.save(model.state_dict(), save_path)
#
#             # 删除上一个epoch的模型文件（如果存在）
#             if prev_save_path_ema and prev_save_path:
#                 os.remove(prev_save_path_ema)
#                 os.remove(prev_save_path)
#
#             # 更新上一个保存路径
#             prev_save_path_ema = save_path_ema
#             prev_save_path = save_path
#
#             # 计数器(map50有增高则重置为0)
#             no_improvement_count = 0
#
#             # 记录详细验证结果到日志
#             with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:
#                 f.write("=" * 50 + "\n")
#                 f.write("EPOCH {} VALIDATION RESULTS (NEW BEST!):\n".format(cur_epoch))
#                 f.write("=" * 50 + "\n")
#                 f.write("True Positives: {}\n".format(eval_results['tp']))
#                 f.write("False Positives: {}\n".format(eval_results['fp']))
#                 f.write("Mean Precision: {:.6f}\n".format(eval_results['m_pre']))
#                 f.write("Mean Recall: {:.6f}\n".format(eval_results['m_rec']))
#                 f.write("mAP@0.5: {:.6f} (BEST!)\n".format(eval_results['map50']))
#                 f.write("mAP@0.5:0.95: {:.6f}\n".format(eval_results['mean_ap']))
#                 f.write("Previous best mAP@0.5: {:.6f}\n".format(best_map50 if cur_epoch > 1 else 0))
#                 f.write("Improvement: {:.6f}\n".format(eval_results['map50'] - (best_map50 if cur_epoch > 1 else 0)))
#                 f.write("=" * 50 + "\n\n")
#
#         else:
#             no_improvement_count += 1
#
#             # 记录验证结果到日志（没有改进的情况）
#             with open(os.path.join(config['save_folder'], "logging.txt"), "a") as f:
#                 f.write("=" * 50 + "\n")
#                 f.write("EPOCH {} VALIDATION RESULTS:\n".format(cur_epoch))
#                 f.write("=" * 50 + "\n")
#                 f.write("True Positives: {}\n".format(eval_results['tp']))
#                 f.write("False Positives: {}\n".format(eval_results['fp']))
#                 f.write("Mean Precision: {:.6f}\n".format(eval_results['m_pre']))
#                 f.write("Mean Recall: {:.6f}\n".format(eval_results['m_rec']))
#                 f.write("mAP@0.5: {:.6f}\n".format(eval_results['map50']))
#                 f.write("mAP@0.5:0.95: {:.6f}\n".format(eval_results['mean_ap']))
#                 f.write("Best mAP@0.5 so far: {:.6f}\n".format(best_map50))
#                 f.write("No improvement count: {}/100\n".format(no_improvement_count))
#                 f.write("=" * 50 + "\n\n")
#
#         # 验证完成后重新设置模型为训练模式
#         model.train()
#         print("Current epoch {} completed. Best map50: {}".format(cur_epoch, best_map50), flush=True)
#         cur_epoch += 1
#
#
# @torch.no_grad()
# def eval(config, model):
#     dataset = build_dataset(config, phase='test')
#     # 若cuda显存不足，可自己设置batchsize，默认32.
#     dataloader = data.DataLoader(dataset, 2, False, collate_fn=collate_fn
#                                  , num_workers=2, pin_memory=True)
#     # 直接接收训练后的model进行验证，不再使用build_yow0v3(config)初始化加载模型
#     model.eval()
#     model.to("cuda")
#
#     # Configure
#     # iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
#     iou_v = torch.tensor([0.5]).cuda()
#     n_iou = iou_v.numel()
#
#     metrics = []
#     p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
#
#     total_detections = 0
#     total_ground_truths = 0
#
#     for batch_clip, batch_bboxes, batch_labels in p_bar:
#         batch_clip = batch_clip.to("cuda")
#
#         targets = []
#         for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
#             target = torch.Tensor(bboxes.shape[0], 6)
#             target[:, 0] = i
#             target[:, 1] = labels
#             target[:, 2:] = bboxes
#             targets.append(target)
#             total_ground_truths += bboxes.shape[0]  # 统计真实标签数量
#
#         targets = torch.cat(targets, dim=0).to("cuda")
#
#         height = config['img_size']
#         width = config['img_size']
#
#         # Inference
#         outputs = model(batch_clip)
#
#         # NMS
#         targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
#         outputs = non_max_suppression(outputs, 0.5, 0.5)
#
#         # Metrics
#         for i, output in enumerate(outputs):
#             labels = targets[targets[:, 0] == i, 1:]
#
#             if output.shape[0] > 0:
#                 total_detections += output.shape[0]  # 统计检测结果数量
#
#             if output.shape[0] == 0:
#                 # 如果没有检测结果但有真实标签，添加空的metrics
#                 if labels.shape[0]:
#                     correct = torch.zeros((0, n_iou), dtype=torch.bool).cuda()
#                     conf = torch.zeros(0).cuda()
#                     pred_cls = torch.zeros(0).cuda()
#                     target_cls = labels[:, 0].cuda()
#                     metrics.append((correct, conf, pred_cls, target_cls))
#                 continue
#
#             detections = output.clone()
#             correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
#
#             # Evaluate
#             if labels.shape[0]:
#                 tbox = labels[:, 1:5].clone()  # target boxes
#
#                 t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
#                 iou = box_iou(t_tensor[:, 1:], detections[:, :4])
#                 correct_class = t_tensor[:, 0:1] == detections[:, 5]
#
#                 for j in range(len(iou_v)):
#                     x = torch.where((iou >= iou_v[j]) & correct_class)
#                     if x[0].shape[0]:
#                         matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
#                         matches = matches.cpu().numpy()
#                         if x[0].shape[0] > 1:
#                             matches = matches[matches[:, 2].argsort()[::-1]]
#                             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#                         correct[matches[:, 1].astype(int), j] = True
#
#             # 收集metrics数据，格式匹配compute_ap函数
#             metrics.append((
#                 correct,  # True positives (shape: [n_detections, n_iou])
#                 output[:, 4],  # Confidence scores
#                 output[:, 5],  # Predicted classes
#                 labels[:, 0] if labels.shape[0] > 0 else torch.tensor([]).cuda()  # Target classes
#             ))
#
#     # Compute metrics - 重新组织数据格式
#     if len(metrics) > 0:
#         # 分别收集所有批次的数据
#         all_correct = []
#         all_conf = []
#         all_pred_cls = []
#         all_target_cls = []
#
#         for correct, conf, pred_cls, target_cls in metrics:
#             all_correct.append(correct.cpu().numpy())
#             all_conf.append(conf.cpu().numpy())
#             all_pred_cls.append(pred_cls.cpu().numpy())
#             all_target_cls.append(target_cls.cpu().numpy())
#
#         # 合并所有数据
#         tp = np.concatenate(all_correct, axis=0) if len(all_correct) > 0 else np.array([]).reshape(0, n_iou)
#         conf = np.concatenate(all_conf, axis=0) if len(all_conf) > 0 else np.array([])
#         pred_cls = np.concatenate(all_pred_cls, axis=0) if len(all_pred_cls) > 0 else np.array([])
#         target_cls = np.concatenate(all_target_cls, axis=0) if len(all_target_cls) > 0 else np.array([])
#     #
#     #     # 计算AP
#     #     if len(tp) > 0 and len(target_cls) > 0:
#     #         tp_count, fp_count, m_pre, m_rec, map50, mean_ap = compute_ap(tp, conf, pred_cls, target_cls)
#     #     else:
#     #         tp_count, fp_count, m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0, 0, 0
#     # else:
#     #     tp_count, fp_count, m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0, 0, 0
#         if len(tp) > 0 and len(target_cls) > 0:
#             tp_count, fp_count, m_pre, m_rec, map50, mean_ap, ap_classes, ap_values = compute_ap(tp, conf, pred_cls, target_cls)
#         else:
#             tp_count, fp_count, m_pre, m_rec, map50, mean_ap = 0, 0, 0, 0, 0, 0
#             ap_classes, ap_values = [], []
#
#     # Print detailed results
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
#     # 写入日志文件
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
#     # Return results as dictionary for better handling
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
#     # Return results
#     model.float()  # for training
#     return results
#
#
# def smooth(y, f=0.05):
#     # Box filter of fraction f
#     nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
#     p = numpy.ones(nf // 2)  # ones padding
#     yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
#     return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed
#
#
# def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
#     """
#     Compute the average precision, given the recall and precision curves.
#     Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
#     # Arguments
#         tp:  True positives (nparray, nx1 or nx10).
#         conf:  Object-ness value from 0-1 (nparray).
#         pred_cls:  Predicted object classes (nparray).
#         target_cls:  True object classes (nparray).
#     # Returns
#         The average precision
#     """
#     # Sort by object-ness
#     i = numpy.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
#
#     # Find unique classes
#     unique_classes, nt = numpy.unique(target_cls, return_counts=True)
#     nc = unique_classes.shape[0]  # number of classes, number of detections
#
#     # Create Precision-Recall curve and compute AP for each class
#     p = numpy.zeros((nc, 1000))
#     r = numpy.zeros((nc, 1000))
#     ap = numpy.zeros((nc, tp.shape[1]))
#     px, py = numpy.linspace(0, 1, 1000), []  # for plotting
#     for ci, c in enumerate(unique_classes):
#         i = pred_cls == c
#         nl = nt[ci]  # number of labels
#         no = i.sum()  # number of outputs
#         if no == 0 or nl == 0:
#             continue
#
#         # Accumulate FPs and TPs
#         fpc = (1 - tp[i]).cumsum(0)
#         tpc = tp[i].cumsum(0)
#
#         # Recall
#         recall = tpc / (nl + eps)  # recall curve
#         # negative x, xp because xp decreases
#         r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)
#
#         # Precision
#         precision = tpc / (tpc + fpc)  # precision curve
#         p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score
#
#         # AP from recall-precision curve
#         for j in range(tp.shape[1]):
#             m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
#             m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))
#
#             # Compute the precision envelope
#             m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))
#
#             # Integrate area under curve
#             # x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
#             # ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
#             ap[ci, j] = numpy.trapz(m_pre, m_rec)  # integrate
#
#     # Compute F1 (harmonic mean of precision and recall)
#     f1 = 2 * p * r / (p + r + eps)
#
#     i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
#     p, r, f1 = p[:, i], r[:, i], f1[:, i]
#     tp = (r * nt).round()  # true positives
#     fp = (tp / (p + eps) - tp).round()  # false positives
#     ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#     m_pre, m_rec = p.mean(), r.mean()
#     map50, mean_ap = ap50.mean(), ap.mean()
#     return tp, fp, m_pre, m_rec, map50, mean_ap, unique_classes, ap50

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
import torchvision.transforms.functional as FT  # 未用到也可删除
import torch.optim as optim
import tqdm
from PIL import Image

from utils.gradflow_check import plot_grad_flow  # 未用到也可删除
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
from cus_datasets.ucf.transforms import UCF_transform  # 未用到也可删除


# -------------------- 小工具 --------------------
def _parse_img_size(img_size):
    """返回 (H, W) 元组；支持 int / (H, W) / [H, W]"""
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise TypeError(f"img_size 格式不支持：{img_size!r}，请用 int 或 (H, W)")

def _to_cuda_tensor(x, device, is_label=False, num_classes=None):
    """
    x: numpy / list / torch.Tensor
    is_label: True 则根据维度与 dtype 决定 long/float；并可在训练阶段转 one-hot
    num_classes: 仅当训练阶段需要把 id -> one-hot 时使用
    """
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)

    # 对标签的稳健化（训练/验证各自调用时传参不同）
    if is_label:
        # 如果明确是类别 id（1D 或 Nx1 或 整型），则 long
        if t.ndim == 1 or (t.ndim == 2 and t.shape[-1] == 1) or t.dtype in (torch.int32, torch.int64):
            t = t.view(-1).long()
            # 若给了 num_classes，则把 id 转 one-hot（训练阶段使用）
            if num_classes is not None:
                t = F.one_hot(t, num_classes=num_classes).float()
        else:
            # 视作 one-hot
            t = t.float()
    else:
        # bbox 均按 float
        t = t.float()

    return t.to(device, non_blocking=True)


# -------------------- 训练主函数 --------------------
def train_model(config):
    # 保存配置文件
    source_file = config['config_path']
    destination_file = os.path.join(config['save_folder'], 'config.yaml')
    os.makedirs(config['save_folder'], exist_ok=True)
    shutil.copyfile(source_file, destination_file)

    # 数据与模型
    dataset = build_dataset(config, phase='train')
    dataloader = data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model = build_yowov3(config)
    get_info(config, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    criterion = build_loss(model, config)

    # 优化器
    g = [], [], []  # 带衰减、不带衰减的 weight 与 bias
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

    # 追踪最佳 mAP@0.5
    global best_map50, no_improvement_count
    best_map50 = 0.0
    no_improvement_count = 0

    prev_save_path_ema = None
    prev_save_path = None

    # 早停：连续 100 个 epoch 没提升则停止
    while (cur_epoch <= max_epoch) and (no_improvement_count < 100):
        cnt_pram_update = 0
        for iteration, (batch_clip, batch_bboxes, batch_labels) in enumerate(dataloader):
            batch_size = batch_clip.shape[0]
            batch_clip = batch_clip.to(device, non_blocking=True)

            # 统一转 tensor 并搬到 GPU（修复 numpy .to 报错）
            batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
            # 训练阶段：labels 若是 id，则转 one-hot
            batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=num_classes) for l in batch_labels]

            outputs = model(batch_clip)

            # 组装 targets: [img_idx, x1, y1, x2, y2, one-hot...]
            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                nbox = bboxes.shape[0]
                if nbox == 0:
                    continue
                if labels.ndim == 1:
                    # 已在 _to_cuda_tensor 转成 one-hot；这里只是兜底
                    labels = F.one_hot(labels.long(), num_classes=num_classes).float()
                elif labels.ndim == 2 and labels.shape[1] != num_classes:
                    # 兜底：若维度不符也转一下
                    labels = F.one_hot(labels.view(-1).long(), num_classes=num_classes).float()

                nclass = labels.shape[1]
                tgt = torch.zeros(nbox, 5 + nclass, device=device, dtype=torch.float32)
                tgt[:, 0] = i
                tgt[:, 1:5] = bboxes
                tgt[:, 5:] = labels
                targets.append(tgt)

            if len(targets) == 0:
                # 无目标的 batch，直接跳过累计（避免除 0）
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

        if cur_epoch in adjustlr_schedule:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

        # 验证
        eval_results = eval(config, model)

        # 保存最佳
        if cur_epoch == 1:
            save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
            torch.save(ema.ema.state_dict(), save_path_ema)
            prev_save_path_ema = save_path_ema

            save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
            torch.save(model.state_dict(), save_path)
            prev_save_path = save_path

            best_map50 = eval_results['map50']

            with open(os.path.join(save_folder, "logging.txt"), "a") as f:
                f.write("=" * 50 + "\n")
                f.write(f"EPOCH {cur_epoch} VALIDATION RESULTS:\n")
                f.write("=" * 50 + "\n")
                f.write(f"True Positives: {eval_results['tp']}\n")
                f.write(f"False Positives: {eval_results['fp']}\n")
                f.write(f"Mean Precision: {eval_results['m_pre']:.6f}\n")
                f.write(f"Mean Recall: {eval_results['m_rec']:.6f}\n")
                f.write(f"mAP@0.5: {eval_results['map50']:.6f}\n")
                f.write(f"mAP@0.5:0.95: {eval_results['mean_ap']:.6f}\n")
                f.write(f"Best mAP@0.5 so far: {best_map50:.6f}\n")
                f.write("=" * 50 + "\n\n")

        elif eval_results['map50'] > best_map50:
            prev_best = best_map50
            best_map50 = eval_results['map50']
            print(f"New best model found at epoch {cur_epoch} with map50: {best_map50}", flush=True)

            # 保存当前最佳（EMA 和原模型）
            save_path_ema = os.path.join(save_folder, f"ema_epoch_{cur_epoch}.pth")
            torch.save(ema.ema.state_dict(), save_path_ema)
            save_path = os.path.join(save_folder, f"epoch_{cur_epoch}.pth")
            torch.save(model.state_dict(), save_path)

            # 删除上一个 epoch 的权重
            if prev_save_path_ema and os.path.isfile(prev_save_path_ema):
                os.remove(prev_save_path_ema)
            if prev_save_path and os.path.isfile(prev_save_path):
                os.remove(prev_save_path)

            prev_save_path_ema = save_path_ema
            prev_save_path = save_path
            no_improvement_count = 0

            with open(os.path.join(save_folder, "logging.txt"), "a") as f:
                f.write("=" * 50 + "\n")
                f.write(f"EPOCH {cur_epoch} VALIDATION RESULTS (NEW BEST!):\n")
                f.write("=" * 50 + "\n")
                f.write(f"True Positives: {eval_results['tp']}\n")
                f.write(f"False Positives: {eval_results['fp']}\n")
                f.write(f"Mean Precision: {eval_results['m_pre']:.6f}\n")
                f.write(f"Mean Recall: {eval_results['m_rec']:.6f}\n")
                f.write(f"mAP@0.5: {eval_results['map50']:.6f} (BEST!)\n")
                f.write(f"mAP@0.5:0.95: {eval_results['mean_ap']:.6f}\n")
                f.write(f"Previous best mAP@0.5: {prev_best:.6f}\n")
                f.write(f"Improvement: {best_map50 - prev_best:.6f}\n")
                f.write("=" * 50 + "\n\n")

        else:
            no_improvement_count += 1
            with open(os.path.join(save_folder, "logging.txt"), "a") as f:
                f.write("=" * 50 + "\n")
                f.write(f"EPOCH {cur_epoch} VALIDATION RESULTS:\n")
                f.write("=" * 50 + "\n")
                f.write(f"True Positives: {eval_results['tp']}\n")
                f.write(f"False Positives: {eval_results['fp']}\n")
                f.write(f"Mean Precision: {eval_results['m_pre']:.6f}\n")
                f.write(f"Mean Recall: {eval_results['m_rec']:.6f}\n")
                f.write(f"mAP@0.5: {eval_results['map50']:.6f}\n")
                f.write(f"mAP@0.5:0.95: {eval_results['mean_ap']:.6f}\n")
                f.write(f"Best mAP@0.5 so far: {best_map50:.6f}\n")
                f.write(f"No improvement count: {no_improvement_count}/100\n")
                f.write("=" * 50 + "\n\n")

        model.train()
        print(f"Current epoch {cur_epoch} completed. Best map50: {best_map50}", flush=True)
        cur_epoch += 1


# -------------------- 验证 --------------------
@torch.no_grad()
def eval(config, model):
    dataset = build_dataset(config, phase='test')
    dataloader = data.DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    # 仅 mAP@0.5
    iou_v = torch.tensor([0.5], device=device)
    n_iou = iou_v.numel()

    metrics = []
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    total_detections = 0
    total_ground_truths = 0

    H, W = _parse_img_size(config['img_size'])

    for batch_clip, batch_bboxes, batch_labels in p_bar:
        batch_clip = batch_clip.to(device, non_blocking=True)

        # 统一类型与设备
        batch_bboxes = [_to_cuda_tensor(b, device, is_label=False) for b in batch_bboxes]
        # 验证阶段 labels 需要 id；若是 one-hot 会在下面转 id
        batch_labels = [_to_cuda_tensor(l, device, is_label=True, num_classes=None) for l in batch_labels]

        # 组装 targets: [img_idx, cls_id, x1, y1, x2, y2]
        targets = []
        for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            n = bboxes.shape[0]
            if n == 0:
                continue

            # labels: 若是 one-hot -> argmax；否则直接视作 id
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

        if len(targets) == 0:
            # 没有目标的 batch，直接推理、跳过匹配
            outputs = model(batch_clip)
            outputs = non_max_suppression(outputs, 0.5, 0.5)
            continue

        targets = torch.cat(targets, dim=0)

        # 推理
        outputs = model(batch_clip)

        # 还原到像素坐标
        targets[:, 2:] *= torch.tensor((W, H, W, H), device=device, dtype=torch.float32)

        # NMS
        outputs = non_max_suppression(outputs, 0.5, 0.5)

        # 统计 metrics
        for i, output in enumerate(outputs):
            labels = targets[targets[:, 0] == i, 1:]

            if output is not None and output.shape[0] > 0:
                total_detections += output.shape[0]
            else:
                # 若无检测但有 GT，则记录空匹配
                if labels.shape[0] > 0:
                    correct = torch.zeros((0, n_iou), dtype=torch.bool, device=device)
                    conf = torch.zeros(0, device=device)
                    pred_cls = torch.zeros(0, device=device)
                    target_cls = labels[:, 0].to(device)
                    metrics.append((correct, conf, pred_cls, target_cls))
                continue

            detections = output.clone()
            correct = torch.zeros(detections.shape[0], n_iou, dtype=torch.bool, device=device)

            if labels.shape[0] > 0:
                tbox = labels[:, 1:5].clone()  # xyxy
                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)  # [cls, x1,y1,x2,y2]
                iou = box_iou(t_tensor[:, 1:], detections[:, :4])  # [n_gt, n_det]
                correct_class = t_tensor[:, 0:1] == detections[:, 5]

                for j in range(n_iou):
                    x = torch.where((iou >= iou_v[j]) & correct_class)
                    if x[0].numel():
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)  # [k, 3]
                        matches = matches.detach().cpu().numpy()
                        if matches.shape[0] > 1:
                            # 先按 IoU 降序，再做唯一
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), j] = True

            metrics.append((
                correct,
                detections[:, 4],
                detections[:, 5],
                labels[:, 0] if labels.shape[0] > 0 else torch.tensor([], device=device)
            ))

    # 汇总指标
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
            tp_count, fp_count, m_pre, m_rec, map50, mean_ap, ap_classes, ap_values = compute_ap(tp, conf, pred_cls, target_cls)
        else:
            tp_count = fp_count = 0
            m_pre = m_rec = map50 = mean_ap = 0.0
            ap_classes, ap_values = [], []
    else:
        tp_count = fp_count = 0
        m_pre = m_rec = map50 = mean_ap = 0.0
        ap_classes, ap_values = [], []

    # 打印
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

    # 记录
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

    model.float()  # back to train-friendly dtype
    return results


# -------------------- 计算 AP --------------------
def smooth(y, f=0.05):
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')

def compute_ap(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    计算 AP，tp: [N, n_iou] 的布尔矩阵；只用到 n_iou=1（即0.5）的一列时也兼容。
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
