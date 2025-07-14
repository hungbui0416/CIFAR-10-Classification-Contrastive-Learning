import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from data_utils import MultiTransform, train_val_split, TransformedDataset
from networks import CNN, ResNet18, ViT
from contrastive_loss import ContrastiveLoss
from train_test_utils import cl_train, cl_epoch_log

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-views', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--network', type=str, default='cnn')
    parser.add_argument('--projection-dim', type=int, default=128)

    parser.add_argument('--contrast-mode', type=str, default='scl')
    parser.add_argument('--temperature', type=int, default=0.1)
    
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=1000)

    parser.add_argument('--log-freq', type=int, default=10)
    return parser.parse_args()

def main():
    args = arg_parser()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    train_dataset, _ = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, MultiTransform(transform, num_transforms=args.num_views))

    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    model_dir = f'./ckpts/cl_{args.network}.pth'

    if args.network == 'cnn':
        model = CNN(num_classes=args.projection_dim).to(device)
    elif args.network == 'resnet18':
        model = ResNet18(num_classes=args.projection_dim).to(device)
    elif args.network == 'vit':
        model = ViT(num_classes=args.projection_dim).to(device)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    model = nn.DataParallel(model)
    
    criterion = ContrastiveLoss(contrast_mode=args.contrast_mode, temperature=args.temperature)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    wandb.login()
    wandb.init(project='CIFAR-10-Contrastive-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    for epoch in range(1, args.num_epochs + 1):
        train_loss = cl_train(epoch, loader, device, model, criterion, optimizer, scheduler, args.log_freq)
        cl_epoch_log(epoch, args.num_epochs, train_loss)

    torch.save(model.module.state_dict(), model_dir)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == '__main__':
    main()