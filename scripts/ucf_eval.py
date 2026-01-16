import torch
import torch.utils.data as data
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as FT
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
import time
import xml.etree.ElementTree as ET
import os
import cv2
import random
import sys
import glob

from math import sqrt

from cus_datasets.build_dataset import build_dataset
from cus_datasets.collate_fn import collate_fn
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from utils.box import non_max_suppression, box_iou
from evaluator.eval import compute_ap
import tqdm
from cus_datasets.ucf.transforms import UCF_transform
from utils.flops import get_info

@torch.no_grad()
def eval(config):

    ###############################################
    dataset = build_dataset(config, phase='test')

    dataloader = data.DataLoader(dataset, 8, False, collate_fn=collate_fn
                                 , num_workers=8, pin_memory=True)

    model = build_yowov3(config)
    get_info(config, model)
    model.to("cuda")
    model.eval()
    ###############################################

    # Configure
    #iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    iou_v = torch.tensor([0.5]).cuda()
    n_iou = iou_v.numel()

    m_pre = 0.
    m_rec = 0.
    map50 = 0.
    mean_ap = 0.
    metrics = []
    p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))

    with torch.no_grad():
        for batch_clip, batch_bboxes, batch_labels in p_bar:
            batch_clip = batch_clip.to("cuda")

            targets = []
            for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
                target = torch.Tensor(bboxes.shape[0], 6)
                target[:, 0] = i
                target[:, 1] = labels
                target[:, 2:] = bboxes
                targets.append(target)

            targets = torch.cat(targets, dim=0).to("cuda")

            height = config['img_size']
            width  = config['img_size']

            # Inference
            outputs = model(batch_clip)

            # NMS
            targets[:, 2:] *= torch.tensor((width, height, width, height)).cuda()  # to pixels
            outputs = non_max_suppression(outputs, 0.005, 0.5)

            # Metrics
            for i, output in enumerate(outputs):
                labels = targets[targets[:, 0] == i, 1:]
                correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if labels.shape[0]:
                        metrics.append((correct, *torch.zeros((3, 0)).cuda()))
                    continue

                detections = output.clone()
                #util.scale(detections[:, :4], samples[i].shape[1:], shapes[i][0], shapes[i][1])

                # Evaluate
                if labels.shape[0]:
                    tbox = labels[:, 1:5].clone()  # target boxes
                    #tbox[:, 0] = labels[:, 1] - labels[:, 3] / 2  # top left x
                    #tbox[:, 1] = labels[:, 2] - labels[:, 4] / 2  # top left y
                    #tbox[:, 2] = labels[:, 1] + labels[:, 3] / 2  # bottom right x
                    #tbox[:, 3] = labels[:, 2] + labels[:, 4] / 2  # bottom right y
                    #util.scale(tbox, samples[i].shape[1:], shapes[i][0], shapes[i][1])

                    correct = np.zeros((detections.shape[0], iou_v.shape[0]))
                    correct = correct.astype(bool)

                    t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                    iou = box_iou(t_tensor[:, 1:], detections[:, :4])
                    correct_class = t_tensor[:, 0:1] == detections[:, 5]
                    for j in range(len(iou_v)):
                        x = torch.where((iou >= iou_v[j]) & correct_class)
                        if x[0].shape[0]:
                            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1)
                            matches = matches.cpu().numpy()
                            if x[0].shape[0] > 1:
                                matches = matches[matches[:, 2].argsort()[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), j] = True
                    correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
                metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

        # Compute metrics
        metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        if len(metrics) and metrics[0].any():
            tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics)

        # Print results
        print('%10.3g' * 3 % (m_pre, m_rec, mean_ap), flush=True)

        # Return results
        model.float()  # for training
        #return map50, mean_ap
        print(map50, flush=True)
        print(flush=True)
        print("=================================================================", flush=True)
        print(flush=True)
        print(mean_ap, flush=True)

#
# import torch
# import torch.utils.data as data
# import tqdm
# import numpy as np
# from utils.box import non_max_suppression, box_iou
# from evaluator.eval import compute_ap
# from cus_datasets.build_dataset import build_dataset
# from cus_datasets.collate_fn import collate_fn
# from model.TSN.YOWOv3 import build_yowov3
# from utils.flops import get_info
# from datetime import datetime
#
#
# @torch.no_grad()
# def eval(config):
#     dataset = build_dataset(config, phase='test')
#     dataloader = data.DataLoader(dataset, 2, False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
#
#     model = build_yowov3(config)
#     get_info(config, model)
#     model.to("cuda")
#     model.eval()
#
#     iou_v = torch.tensor([0.5]).cuda()
#     n_iou = iou_v.numel()
#
#     metrics = []
#     p_bar = tqdm.tqdm(dataloader, desc=('%10s' * 3) % ('precision', 'recall', 'mAP'))
#
#     with torch.no_grad():
#         for batch_clip, batch_bboxes, batch_labels in p_bar:
#             if batch_clip.ndim == 4:
#                 batch_clip = batch_clip.unsqueeze(1)  # Add temporal dim
#             batch_clip = batch_clip.to("cuda")
#
#             targets = []
#             for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
#                 target = torch.zeros((bboxes.shape[0], 6))
#                 target[:, 0] = i
#                 target[:, 1] = labels
#                 target[:, 2:] = bboxes
#                 targets.append(target)
#             targets = torch.cat(targets, dim=0).to("cuda")
#
#             H, W = config['img_size'], config['img_size']
#             targets[:, 2:] *= torch.tensor([W, H, W, H], device=targets.device)
#
#             outputs = model(batch_clip)
#             outputs = non_max_suppression(outputs, conf_threshold=0.005, iou_threshold=0.5)
#
#             for i, output in enumerate(outputs):
#                 labels = targets[targets[:, 0] == i, 1:]
#                 correct = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()
#
#                 if output.shape[0] == 0:
#                     if labels.shape[0]:
#                         metrics.append((correct, *torch.zeros((3, 0)).cuda()))
#                     continue
#
#                 detections = output.clone()
#
#                 if labels.shape[0]:
#                     tbox = labels[:, 1:5].clone()
#                     t_tensor = torch.cat((labels[:, 0:1], tbox), dim=1)
#
#                     iou = box_iou(t_tensor[:, 1:], detections[:, :4])
#                     correct_class = t_tensor[:, 0:1] == detections[:, 5].long().unsqueeze(0)
#
#                     correct = np.zeros((detections.shape[0], iou_v.numel()), dtype=bool)
#                     for j in range(iou_v.numel()):
#                         x = torch.where((iou >= iou_v[j]) & correct_class)
#                         if x[0].numel():
#                             matches = torch.cat([torch.stack(x, 1), iou[x[0], x[1]][:, None]], 1)
#                             matches = matches.cpu().numpy()
#                             if matches.shape[0] > 1:
#                                 matches = matches[matches[:, 2].argsort()[::-1]]
#                                 matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                                 matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#                             correct[matches[:, 1].astype(int), j] = True
#                     correct = torch.tensor(correct, dtype=torch.bool, device="cuda")
#
#                 metrics.append((correct, output[:, 4], output[:, 5], labels[:, 0]))
#
#     metrics = [torch.cat(x, 0).cpu().numpy() for x in zip(*metrics)]
#     if len(metrics) and metrics[0].any():
#         tp, conf, m_pre, m_rec, map50, mean_ap, ap_class = compute_ap(*metrics)
#     else:
#         tp = conf = map50 = mean_ap = m_pre = m_rec = 0.
#         ap_class = []
#
#     # 输出日志
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     output_log = [f"Evaluation Results - {timestamp}", "=" * 50]
#     output_log += [
#         f"{'Precision:':<20} {m_pre:.4f}",
#         f"{'Recall:':<20} {m_rec:.4f}",
#         f"{'mAP@0.5:':<20} {map50:.4f}",
#         f"{'mAP@[.5:.95]:':<20} {mean_ap:.4f}",
#         ""
#     ]
#
#     output_log.append("Per-Class AP:")
#     class_names = config.get('class_name', [])
#     for cls_id, ap in enumerate(ap_class):
#         class_str = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id:02d}"
#         output_log.append(f"{class_str:<20s} : AP = {ap:.4f}")
#
#     for line in output_log:
#         print(line)
#     with open("eval_log.txt", "w") as f:
#         f.write('\n'.join(output_log) + '\n')
#
#     model.float()
#     print("\nLog written to eval_log.txt")
# import numpy as np
#
# def compute_ap(tp, conf, pred_cls, target_cls):
#     i = np.argsort(-conf)
#     tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
#
#     unique_classes = np.unique(np.concatenate((pred_cls, target_cls)))
#     ap_class = []
#     precisions = []
#     recalls = []
#
#     for c in unique_classes:
#         i = pred_cls == c
#         n_gt = (target_cls == c).sum()
#         n_p = i.sum()
#         if n_gt == 0 or n_p == 0:
#             ap_class.append(0.0)
#             continue
#
#         fpc = (1 - tp[i]).cumsum()
#         tpc = tp[i].cumsum()
#         recall_curve = tpc / (n_gt + 1e-16)
#         precision_curve = tpc / (tpc + fpc)
#
#         recalls.append(recall_curve[-1])
#         precisions.append(precision_curve[-1])
#         ap = compute_ap_from_pr_curve(recall_curve, precision_curve)
#         ap_class.append(ap)
#
#     mean_precision = np.mean(precisions) if precisions else 0.
#     mean_recall = np.mean(recalls) if recalls else 0.
#     map50 = np.mean(ap_class)
#     mean_ap = map50
#
#     return tp, conf, mean_precision, mean_recall, map50, mean_ap, ap_class
#
#
# def compute_ap_from_pr_curve(recall, precision):
#     mrec = np.concatenate(([0.0], recall, [1.0]))
#     mpre = np.concatenate(([0.0], precision, [0.0]))
#     for i in range(mpre.size - 1, 0, -1):
#         mpre[i - 1] = max(mpre[i - 1], mpre[i])
#     idx = np.where(mrec[1:] != mrec[:-1])[0]
#     return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

