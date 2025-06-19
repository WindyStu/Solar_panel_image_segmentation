from torch.utils.tensorboard import SummaryWriter
from pathlib import Path


class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag, images, step):
        self.writer.add_images(tag, images, step)

    def close(self):
        self.writer.close()