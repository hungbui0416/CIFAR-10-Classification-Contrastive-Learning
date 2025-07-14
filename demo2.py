import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import os

from networks import CNN, ResNet18, ViT
from data_utils import get_misclassified_images, save_misclassified_images

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='cnn')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='test')

    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()

def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    normalize = transforms.Normalize(mean=mean, std=std)
    denormalize = transforms.Normalize(mean=-mean/std, std=1/std)

    tensor_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    pil_transform = transforms.Compose([
        denormalize,
        transforms.ToPILImage(),
    ])

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
    if args.network == 'cnn':
        model = CNN(num_classes=10).to(device)
    elif args.network == 'resnet18':
        model = ResNet18(num_classes=10).to(device)
    elif args.network == 'vit':
        model = ViT(num_classes=10).to(device)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    if args.dataset == 'train':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=tensor_transform)
    elif args.dataset == 'test':
        dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tensor_transform)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    misclassified_dict = get_misclassified_images(model, loader, pil_transform)

    save_misclassified_images(misclassified_dict, save_dir=f'./misclassified/{args.network}_{args.dataset}')

if __name__ == '__main__':
    main()