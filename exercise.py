from torchvision.datasets import CIFAR10
train = CIFAR10(root='/mnt/data/cifar10/train', train=True, download=True)
valid = CIFAR10(root='/mnt/data/cifar10/test', train=False, download=True)