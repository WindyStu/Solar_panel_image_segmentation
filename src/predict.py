import argparse
import os
import numpy as np
import yaml
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SolarPanelDataset
from data.transforms import get_val_transforms
# from models.unet import UNet
from models.nested_unet import NestedUNet,U_Net
from utils.postprocess import post_process


class Predictor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

        # 加载模型
        # self.model = UNet(n_channels=3, n_classes=1).to(self.device)
        self.model = U_Net(in_ch=3, out_ch=1).to(self.device)
        self.model.load_state_dict(torch.load(config['model_path']))
        self.model.eval()

        # 测试数据
        self.test_ds = SolarPanelDataset(
            image_dir=config['data']['test_dir'],
            transform=get_val_transforms(config['data']['image_size']),
            mode='test'
        )
        self.test_loader = DataLoader(
            self.test_ds,
            batch_size=config['test']['batch_size'],
            shuffle = False,
            num_workers=4
        )

    def generate_submission(self, output_dir, threshold=0.5):
        """生成符合要求的提交文件"""
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for images, img_names in tqdm(self.test_loader, desc="Generating Predictions"):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = torch.sigmoid(outputs)

                for i in range(outputs.shape[0]):
                    pred = outputs[i].squeeze().cpu().numpy()
                    pred = post_process(pred, threshold=threshold)

                    img_name = img_names[i].replace('image', 'predict')
                    save_path = os.path.join(output_dir, img_name)
                    Image.fromarray(pred).save(save_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test images')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for binary prediction')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置文件
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 合并命令行参数
    config.update({
        'model_path': args.model_path,
        'test_dir': args.test_dir,
        'output_dir': args.output_dir
    })

    # 运行预测
    predictor = Predictor(config)
    predictor.generate_submission(args.output_dir, args.threshold)


if __name__ == "__main__":
    main()