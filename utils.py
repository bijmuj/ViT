from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR
import math

def get_CIFAR100(image_size=(224, 224), batch_size=512, val_batch_size=8):
    """Downloads CIFAR100. The paper used JFT-300M for pretraining, but that dataset is private to Google.

    Args:
        image_size (tuple, optional): Size of images. Defaults to (224, 224).
        batch_size (int, optional): Batch size for training. Defaults to 512.
        val_batch_size (int, optional): Batch size for validation. Defaults to 8.

    Returns:
        DataLoader: Returns training and validation dataloader objects.
    """    
    halves = [0.5, 0.5, 0.5]
    transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(halves, halves)
    ])
    transforms_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(halves, halves)
    ])
    trainset = datasets.CIFAR100(root="./data",
                                train=True,
                                download=True,
                                transform=transforms_train)
    testset = datasets.CIFAR100(root="./data",
                                train=False,
                                download=True,
                                transform=transforms_test)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=RandomSampler(trainset), num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=val_batch_size, sampler=SequentialSampler, pin_memory=True)
    return train_loader, test_loader


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