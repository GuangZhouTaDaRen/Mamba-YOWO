from model.backbone2D import manba

def build_backbone2D(config):
    backbone_2D = config['backbone2D']

    if backbone_2D == 'yolov8':
        backbone2D = manba.build_yolov8(config)

    return backbone2D