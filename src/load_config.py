import yaml


def load_config(path="config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
