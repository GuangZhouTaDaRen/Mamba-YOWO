import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3


# ================== ActivationsAndGradients ==================
class ActivationsAndGradients:
    """提取指定层的 activation 和 gradient"""

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation)
            )
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient
                    )
                )
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient
                    )
                )

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        # 保持与你原来一致：新的梯度插在最前面
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


# ================== GradCAM ==================
class GradCAM:
    def __init__(self,
                 model,
                 target_layers,
                 reshape_transform=None,
                 use_cuda=False):
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform
        )

    @staticmethod
    def get_cam_weights(grads):
        # grads: [N, C, H, W]
        return np.mean(grads, axis=(2, 3), keepdims=True)

    @staticmethod
    def get_loss_classification(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations  # [N,C,H,W]
        cam = weighted_activations.sum(axis=1)  # [N,H,W]
        return cam

    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result

    def compute_cam_per_layer(self, input_tensor):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            cam = self.get_cam_image(layer_activations, layer_grads)
            cam[cam < 0] = 0
            scaled = self.scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled[:, None, :])  # [N,1,H,W]

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def __call__(self, input_tensor, target_category=None, custom_loss=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        # 检测模型用 custom_loss
        if custom_loss is not None:
            loss = custom_loss(output)
        else:
            # 分类场景（保留原逻辑）
            if isinstance(target_category, int):
                target_category = [target_category] * input_tensor.size(0)

            if target_category is None:
                target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
                print(f"category id: {target_category}")
            else:
                assert (len(target_category) == input_tensor.size(0))

            loss = self.get_loss_classification(output, target_category)

        self.model.zero_grad()
        loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor)
        return self.aggregate_multi_layers(cam_per_layer)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


# ================== overlay 函数 ==================
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ================== 辅助函数 ==================
def get_idx2name(config):
    idx2name = config.get("idx2name", None)
    if idx2name is None:
        return None
    fixed = {}
    for k, v in idx2name.items():
        try:
            fixed[int(k)] = v
        except:
            fixed[k] = v
    return fixed


def auto_select_conv_in_fusion(model, clip, loss_fn):
    """
    在 fusion 里从后往前找一个“梯度非零”的 Conv2d 作为 CAM 层，
    避免选到被 detach 掉的层导致全 0。
    """
    device = next(model.parameters()).device
    clip = clip.to(device)

    convs = []
    for m in model.fusion.modules():
        if isinstance(m, nn.Conv2d):
            convs.append(m)

    if not convs:
        raise RuntimeError("model.fusion 中没有 Conv2d，无法做 Grad-CAM")

    chosen = None

    for layer in reversed(convs):
        grads = []

        def fwd_hook(mod, inp, out):
            pass  # 不需要 activation，只看梯度

        def bwd_hook(mod, gin, gout):
            grads.append(gout[0])

        h1 = layer.register_forward_hook(fwd_hook)
        if hasattr(layer, "register_full_backward_hook"):
            h2 = layer.register_full_backward_hook(bwd_hook)
        else:
            h2 = layer.register_backward_hook(bwd_hook)

        model.zero_grad(set_to_none=True)
        out = model(clip)
        loss = loss_fn(out)
        loss.backward(retain_graph=False)

        has_grad = (len(grads) > 0) and (grads[0].abs().sum().item() > 1e-9)

        h1.remove()
        h2.remove()
        torch.cuda.empty_cache()

        if has_grad:
            chosen = layer
            break

    if chosen is None:
        chosen = convs[-1]
        print("WARNING: 所有 fusion Conv2d 梯度几乎为 0，回退到最后一层 Conv2d。")

    return chosen


def build_clip_from_img(img_path, img_size, T=16, device="cuda"):
    """读取一张图片 -> resize -> 构造 (1,3,T,H,W) 伪 clip，并返回 clip 和 [0,1] 的 RGB 图像"""
    img_bgr = cv2.imread(img_path)
    assert img_bgr is not None, f"图片不存在: {img_path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size))

    img_rgb_float = img_rgb.astype(np.float32) / 255.0  # (H,W,3)
    img_tensor = torch.from_numpy(img_rgb_float).permute(2, 0, 1)  # (3,H,W)

    clip = img_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, T, 1, 1)  # (1,3,T,H,W)
    clip = clip.to(device)

    return clip, img_rgb_float  # clip, 原图 [0,1]


# ================== main：对 feeding_no_8527 文件夹所有图片生成整图热力图 ==================
def main():
    # ===== 你指定的 config 与权重 =====
    config_path = "/data/CuiTengPeng/YOWOv3/config/cf/ucf_mamba_Channel.yaml"
    weight_path = "/data/CuiTengPeng/YOWOv3/weights/fig/ucf/mambaOut_shufflenetv2+Channel/ema_epoch_10.pth"

    # RGB 图片目录（和 labels 对应）
    img_dir = "/data/CuiTengPeng/YOWOv3/cus_datasets/ucf24/rgb-images/flying/flying_no_2126"

    save_root = "/data/CuiTengPeng/YOWOv3/CAM/feeding_no_2126_heatmaps"

    os.makedirs(save_root, exist_ok=True)

    # 收集文件夹中的所有图片
    img_paths = [
        os.path.join(img_dir, f)
        for f in sorted(os.listdir(img_dir))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    assert len(img_paths) > 0, f"文件夹中没有图片: {img_dir}"

    # 1. 构建模型
    config = build_config(config_path)
    model = build_yowov3(config)
    ckpt = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(ckpt, strict=False)
    model.cuda().eval()

    idx2name = get_idx2name(config)
    num_classes = int(config["num_classes"])  # 这里应该是 4：creeping / feeding / flying / mating
    img_size = config["img_size"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. 用第一张图构造 clip，自动在 fusion 里选一个“有梯度”的 Conv2d
    first_clip, _ = build_clip_from_img(img_paths[0], img_size, T=16, device=device)

    def det_loss_fn(output):
        """
        output: model(clip) -> [1, N, 5+C]
        我们只关心前 C=num_classes 个行为类：
        0: creeping, 1: feeding, 2: flying, 3: mating
        对所有框、所有这 4 个类的 obj*cls_prob 求和作为 Grad-CAM 的目标，
        让热力图同时覆盖所有这 4 种行为的区域。
        """
        raw = output[0]  # [N, 5+C(+其它?)]
        if raw.numel() == 0:
            return raw.sum() * 0.0

        # 取出 obj 和 4 个行为类别的 logits
        if raw.size(1) < 5 + num_classes:
            raise RuntimeError(f"检测头输出维度 {raw.size(1)} 小于 5+num_classes={5 + num_classes}")

        obj_logit = raw[:, 4]  # [N]
        cls_logits = raw[:, 5:5 + num_classes]  # [N,4] -> creeping/feeding/flying/mating

        # 按 YOLO 风格，用 sigmoid 当概率
        obj_prob = torch.sigmoid(obj_logit)  # [N]
        cls_prob = torch.sigmoid(cls_logits)  # [N,4]

        # 对 4 个行为类都关注：scores = obj * cls_prob
        scores = obj_prob.unsqueeze(1) * cls_prob  # [N,4]

        # 打个简单 log 看一下整体强度（可选）
        max_score, max_idx = scores.view(-1).max(dim=0)
        max_box = max_idx // num_classes
        max_cls = int(max_idx % num_classes)
        if idx2name is not None:
            cls_name = idx2name.get(max_cls, str(max_cls))
        else:
            cls_name = str(max_cls)
        print(
            f"loss_fn (4 behaviors sum): max box={int(max_box)}, cls={max_cls}({cls_name}), score={float(max_score):.3f}")

        # 对所有框 & 4 类的分数求和（或平均），作为 Grad-CAM 的 loss
        loss = scores.sum()
        return loss

    target_layer = auto_select_conv_in_fusion(model, first_clip, det_loss_fn)
    print(f"[Grad-CAM] selected fusion layer = {target_layer}")

    # 3. 构建 GradCAM 提取器（之后复用）
    cam_extractor = GradCAM(model=model, target_layers=[target_layer], use_cuda=(device == "cuda"))

    # 4. 遍历文件夹中所有图片
    for idx, img_path in enumerate(img_paths):
        try:
            clip, img_rgb_float = build_clip_from_img(img_path, img_size, T=16, device=device)
            _, _, _, H, W = clip.shape

            # 计算 Grad-CAM
            grayscale_cam = cam_extractor(
                input_tensor=clip,
                custom_loss=det_loss_fn
            )  # [1,H,W]
            cam_map = grayscale_cam[0, :]  # (H,W) 0~1

            # 叠加整图热力图（无框）
            heatmap_overlay = show_cam_on_image(img_rgb_float, cam_map, use_rgb=True)

            base = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(save_root, f"{base}_heatmap.png")
            cv2.imwrite(save_path, cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))

            if idx % 20 == 0:
                print(f"[{idx}/{len(img_paths)}] saved {save_path}")

        except Exception as e:
            print(f"[{idx}] {img_path} -> Grad-CAM failed: {e}")
            continue

        torch.cuda.empty_cache()

    print("All heatmaps saved to:", save_root)


if __name__ == "__main__":
    main()

