import yaml

def sanitize_cache_name(
    d: dict
):
    """
    Given Argparse, get name of cache for a given setup
    """
    dataset = d['dataset']
    model = d['model']
    model_size = d['model_size']
    layer = d['layer']
    return f"activations/{dataset.split('/')[-1]}_{model}{model_size}_{layer}"

def get_config(config_file):
    if config_file is None: return {}
    return yaml.safe_load(open(config_file, 'r'))