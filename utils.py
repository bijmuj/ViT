from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math
import ml_collections


def load_data(train_dir, val_dir, image_size=(224, 224), batch_size=16, val_batch_size=4):
    halves = [0.5, 0.5, 0.5]
    transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(halves, halves)
    ])
    transforms_val = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(halves, halves)
    ])
    trainset = datasets.ImageFolder(train_dir, transform=transforms_train)
    valset = datasets.ImageFolder(val_dir, transform=transforms_val)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=RandomSampler(trainset), num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=val_batch_size, sampler=SequentialSampler(valset), pin_memory=True)
    return train_loader, val_loader


class WarmupCosineSchedule(LambdaLR):
    """Learning rate scheduler that does linear warmup and cosine decay."""

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        """The actual learning rate updation happens here.

        Args:
            step (int): The step the optimizer is currently on.

        Returns:
            float: Updating scale for the learning rate.
        """        
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def get_config(config):
    C = {'ViT-B_16': get_b16_config(),
        "ViT-B_32": get_b32_config(), 
        "ViT-L_16": get_l16_config(),
        "ViT-L_32": get_l32_config(), 
        "ViT-H_14": get_h14_config()}
    return C[config]


def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = 16
    config.dim = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_hidden = 3072
    config.transformer.heads = 12
    config.transformer.depth = 12
    config.transformer.attention_dropout = 0.0
    config.transformer.dropout = 0.1
    return config


def get_b32_config():
    """Returns the ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches = 32
    return config


def get_l16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = 16
    config.dim = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_hidden = 4096
    config.transformer.heads = 16
    config.transformer.depth = 24
    config.transformer.attention_dropout = 0.0
    config.transformer.dropout = 0.1
    return config


def get_l32_config():
    config = get_l16_config()
    config.patches = 32
    return config


def get_h14_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = 14
    config.dim = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_hidden = 5120
    config.transformer.heads = 16
    config.transformer.depth = 32
    config.transformer.attention_dropout = 0.0
    config.transformer.dropout = 0.1
    return config