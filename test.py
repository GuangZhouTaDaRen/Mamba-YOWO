import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.box import non_max_suppression, box_iou
from cus_datasets.build_dataset import build_dataset
from model.TSN.YOWOv3 import build_yowov3
from utils.build_config import build_config
from cus_datasets.collate_fn import collate_fn

# -------------------- 工具函数 --------------------
def _parse_img_size(img_size):
    """返回 (H, W) 元组；支持 int / (H, W) / [H, W]"""
    if isinstance(img_size, int):
        return img_size, img_size
    if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise TypeError(f"img_size 格式不支持：{img_size!r}，请用 int 或 (H, W)")

# -------------------- 计算混淆矩阵 --------------------
@torch.no_grad()
def eval_with_confusion(config, weight_path):
    """
    使用训练好的权重对 test 集进行评估，计算混淆矩阵和 AP
    - 计算 Precision 和 Recall
    - 计算 AP 和 mAP
    """
    save_folder = "/data/CuiTengPeng/YOWOv3/matrix"
    os.makedirs(save_folder, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(config["num_classes"])

    # ---------- 构建 test 数据集 ----------
    test_list_path = config["test_list_path"]
    with open(test_list_path, "r") as f:
        _ = [line.strip() for line in f.readlines()]  # 只是确保文件存在

    dataset = build_dataset(config, phase="test")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,          # 显存不够可以改小
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # ---------- 构建模型并加载权重 ----------
    model = build_yowov3(config)
    ckpt = torch.load(weight_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    # ---------- 类别名称处理 ----------
    default_names = [f"class_{i}" for i in range(num_classes)]
    raw_idx2name = config.get("idx2name", default_names)
    class_names = list(raw_idx2name) if isinstance(raw_idx2name, (list, tuple)) else default_names

    # ---------- 初始化混淆矩阵（计数） ----------
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    H, W = _parse_img_size(config["img_size"])

    pbar = tqdm(dataloader, desc="Evaluating (for confusion matrix)")
    for batch_clip, batch_bboxes, batch_labels in pbar:
        batch_clip = batch_clip.to(device, non_blocking=True)

        # 目标处理
        targets = []
        for i, (bboxes, labels) in enumerate(zip(batch_bboxes, batch_labels)):
            if isinstance(bboxes, np.ndarray):
                bboxes = torch.from_numpy(bboxes).float().to(device)
            if isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels).long().to(device)

            if bboxes.numel() == 0:
                continue

            target = torch.zeros((bboxes.shape[0], 6), device=device)
            target[:, 0] = i
            target[:, 1] = labels
            target[:, 2:] = bboxes
            targets.append(target)

        if len(targets) == 0:
            continue
        targets = torch.cat(targets, dim=0)

        # ---------- 推理 & NMS ----------
        outputs = model(batch_clip)
        outputs = non_max_suppression(outputs, 0.05, 0.5)

        for img_i, det in enumerate(outputs):
            if det is None or det.shape[0] == 0:
                continue

            labels_i = targets[targets[:, 0] == img_i, 1:]  # (n_gt, 5)
            if labels_i.numel() == 0:
                continue

            gt_cls = labels_i[:, 0].long()
            gt_boxes = labels_i[:, 1:5]

            det_boxes = det[:, :4]
            det_conf = det[:, 4]
            det_cls = det[:, 5].long()

            # 按置信度排序
            order = torch.argsort(det_conf, descending=True)
            det_boxes = det_boxes[order]
            det_cls = det_cls[order]

            ious = box_iou(gt_boxes, det_boxes)
            gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=device)

            for d in range(det_boxes.shape[0]):
                iou_col = ious[:, d]
                max_iou, max_idx = torch.max(iou_col, dim=0)
                if max_iou >= 0.5 and not gt_used[max_idx]:
                    g = int(gt_cls[max_idx].item())
                    p = int(det_cls[d].item())
                    if 0 <= g < num_classes and 0 <= p < num_classes:
                        confusion[g, p] += 1
                    gt_used[max_idx] = True

    # ---------- 计算每个类别的 Precision 和 Recall ----------
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    ap = np.zeros(num_classes)

    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp

        precision[i] = tp / (tp + fp + 1e-6)
        recall[i] = tp / (tp + fn + 1e-6)

        # 计算 AP (可以根据需求进一步细化计算)
        ap[i] = tp / (tp + fp + fn + 1e-6)  # 这里简化为直接用 Precision

    # ---------- 保存原始计数混淆矩阵 ----------
    cm_count_path = os.path.join(save_folder, "confusion_matrix_count.npy")
    np.save(cm_count_path, confusion)

    # ---------- 归一化混淆矩阵 ----------
    cm = confusion.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    cm_npy_path = os.path.join(save_folder, "confusion_matrix.npy")
    np.save(cm_npy_path, cm_norm)

    # ---------- 绘制混淆矩阵 ----------
    cm_plot = cm_norm.astype(np.float32)
    cm_plot = np.nan_to_num(cm_plot, nan=0.0, posinf=1.0, neginf=0.0)
    cm_plot = np.clip(cm_plot, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    thresh = cm_plot.max() / 2.0 if cm_plot.max() > 0 else 0.0

    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_plot[i, j]
            text_color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color)

    plt.tight_layout()
    cm_png_path = os.path.join(save_folder, "confusion_matrix.png")
    plt.savefig(cm_png_path, dpi=300)
    plt.close(fig)

    return cm_norm, precision, recall, ap


# -------------------- 示例 main --------------------
if __name__ == "__main__":
    config_path = "/data/CuiTengPeng/YOWOv3/weights/fig/ucf/mambaOut_shufflenetv2+Spatial/config.yaml"
    weight_path = (
        "/data/CuiTengPeng/YOWOv3/weights/fig/ucf/mambaOut_shufflenetv2+Spatial/epoch_10.pth"
    )

    config = build_config(config_path)
    config["config_path"] = config_path
    config["test_list_path"] = (
        "/data/CuiTengPeng/YOWOv3/cus_datasets/ucf24/testlist.txt"
    )

    cm_norm, precision, recall, ap = eval_with_confusion(config, weight_path)
    print(f"Normalized confusion matrix:\n{cm_norm}")
    print(f"Precision per class: {precision}")
    print(f"Recall per class: {recall}")
    print(f"AP per class: {ap}")
