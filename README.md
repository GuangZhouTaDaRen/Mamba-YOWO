# Mamba-YOWO: A Multi-Behavior Detection Method for Adult Tuta absoluta by Integrating Spatial and Temporal Features


git clone https://github.com/GuangZhouTaDaRen/Mamba-YOWO.git.
```

Use Python Python 3.10, and then download the dependencies:

```powershell
pip install -r requirements.txt
```

### Datasets

#### Tuta absoluta


### Simple command line

We have the following command template:
```powershell
python main.py --mode [mode] --config [config_file_path]
```
Or the shorthand version:

```powershell
python main.py -m [mode] -cf [config_file_path]
```

For ```[mode] = {train, eval, detect, live, onnx}``` for training, evaluation, detection (visualization on the current dataset), live (camera usage) or export to onnx and inference respectively. The```[config_file_path]``` is the path to the config file.

Example of training a model on Tuta absoluta:
```powershell
python main.py --mode train --config /data/CuiTengPeng/YOWOv3/config/cf/ucf_mamba_Channel_clip12.yaml
```



