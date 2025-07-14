import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import argparse
import os
import wandb

from data_utils import TransformedDataset, train_val_split
from networks import CNN, ResNet18, ViT
from train_test_utils import EarlyStopper, sl_train, sl_validate, sl_epoch_log

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--num-workers', type=int, default=4)

    parser.add_argument('--network', type=str, default='cnn')
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--projection-dim', type=int, default=128)
    
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    parser.add_argument('--cosine-annealing', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=500)
    parser.add_argument('--early-stop', action='store_true')
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--min-delta', type=float, default=0.1)

    parser.add_argument('--log-freq', type=int, default=10)
    return parser.parse_args()


def main():
    # parse args
    args = parse_args()
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        val_transform
    ])

    # train dataset
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)

    # train/val split
    train_dataset, val_dataset = train_val_split(dataset, val_ratio=0.2)
    train_set = TransformedDataset(train_dataset, train_transform)
    val_set = TransformedDataset(val_dataset, val_transform)

    # dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model and optimizer
    if not os.path.exists('./ckpts'):
        os.makedirs('./ckpts')
    model_dir = f'./ckpts/sl_{args.network}.pth'

    if args.network == 'cnn':
        if args.model_path:
            model = CNN(num_classes=args.projection_dim).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512*4*4, 10)
            optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            model = CNN(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.network == 'resnet18':
        if args.model_path:
            model = ResNet18(num_classes=args.projection_dim).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            for param in model.parameters():
                param.requires_grad = False
            model.fc = nn.Linear(512, 10)
            optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            model = ResNet18(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.network == 'vit':
        if args.model_path:
            model = ViT(num_classes=args.projection_dim).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            for param in model.parameters():
                param.requires_grad = False
            model.classifier = nn.Linear(64, 10)
            optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            model = ViT(num_classes=10).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown network: {args.network}")
    model = nn.DataParallel(model)

    # criterion, scheduler and early_stopper
    criterion = nn.CrossEntropyLoss()
    if args.cosine_annealing:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    else:
        scheduler = None
    if args.early_stop:
        early_stopper = EarlyStopper(model, model_dir, patience=args.patience, min_delta=args.min_delta)
    else:
        early_stopper = None

    # wandb
    wandb.login()
    wandb.init(project='CIFAR-10-Supervised-Learning')
    wandb.config.update(args)
    wandb.watch(model, log="gradients", log_freq=args.log_freq)

    # train
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = sl_train(epoch, train_loader, device, model, criterion, optimizer, scheduler, args.log_freq)
        val_loss, val_acc = sl_validate(epoch, val_loader, device, model, criterion)
        sl_epoch_log(epoch, args.num_epochs, train_loss, train_acc, val_loss, val_acc)

        if early_stopper and early_stopper.early_stop(val_loss):
            print('Early stop at epoch', epoch)
            break

    # log model to wandb
    if not os.path.exists(model_dir):
        torch.save(model.module.state_dict(), model_dir)
    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(model_dir)
    wandb.log_artifact(artifact, aliases=['latest'])

    wandb.finish()

if __name__ == '__main__':
    main()