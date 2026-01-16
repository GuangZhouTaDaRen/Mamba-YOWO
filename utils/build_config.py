import yaml

def build_config(config_file='config/ucf_config_tiny.yaml'):
    with open(config_file, "r") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    if config['active_checker']:
        pass
    
    return config