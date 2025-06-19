import yaml
from src.train import Trainer



def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    config = load_config("configs/base.yaml")
    trainer = Trainer(config)
    trainer.run()