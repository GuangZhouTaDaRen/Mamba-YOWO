from scripts import train, ava_eval, ucf_eval, detect, live, onnx
import argparse
from utils.build_config import build_config
from model.TSN.YOWOv3 import build_yowov3  # 导入模型构建函数
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOWOv3")

    # 添加命令行参数，mode 参数指定操作类型（训练、评估、实时检测、视频检测、导出ONNX模型）
    parser.add_argument('-m', '--mode', type=str, help='train/eval/live/detect/onnx', required=True)
    parser.add_argument('-cf', '--config', type=str, help='path to config file', required=True)
    parser.add_argument('--video_path', type=str, help='path to the video file (for detect mode)', default=None)

    # 解析命令行参数
    args = parser.parse_args()

    # 加载配置文件
    config = build_config(args.config)

    # 根据 mode 执行不同的任务
    if args.mode == 'train':
        train.train_model(config=config)
    elif args.mode == 'eval':
        # 加载模型
        model = build_yowov3(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 加载预训练模型（如果有的话）
        pretrain_path = config.get('pretrain_path', None)
        if pretrain_path is not None:
            print(f"Loading pre-trained model from {pretrain_path}")
            try:
                model.load_state_dict(torch.load(pretrain_path), strict=False)  # 使用 strict=False 忽略不匹配的键
            except Exception as e:
                print(f"Error loading model state dict: {e}")

        # 根据数据集类型进行评估
        if config['dataset'] in ['ucf', 'jhmdb', 'ucfcrime']:
            train.eval(model=model, config=config)  # 传递模型对象
        elif config['dataset'] == 'ava':
            train.eval(model=model, config=config)  # 传递模型对象
    elif args.mode == 'detect':
        # 需要传递视频路径参数
        if args.video_path is None:
            print("Error: 'video_path' is required for 'detect' mode.")
        else:
            detect.detect_video(config=config, video_path=args.video_path)
    elif args.mode == 'live':
        live.detect(config=config)
    elif args.mode == 'onnx':
        onnx.export2onnx(config=config)
    else:
        print("Invalid mode selected. Please choose from 'train', 'eval', 'live', 'detect', or 'onnx'.")
