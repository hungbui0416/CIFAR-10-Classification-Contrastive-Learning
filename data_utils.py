import torch
from torch.utils.data import Dataset, random_split

import tqdm
import os

class MultiTransform:
    def __init__(self, transform, num_transforms):
        self.transform = transform
        self.num_transforms = num_transforms

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_transforms)]

class TransformedDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y

class HardNegContrastiveDataset(Dataset):
    def __init__(self, dataset, pos_transform, hardneg_transform, hardneg_dict, num_hardnegs, num_randoms, random_transform):
        self.dataset = dataset
        self.pos_transform = pos_transform
        self.hardneg_transform = hardneg_transform
        self.hardneg_dict = hardneg_dict
        self.num_hardnegs = num_hardnegs
        self.num_randoms = num_randoms
        self.random_transform = random_transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]
        pos_imgs = self.pos_transform(anchor_img)
        pos_imgs = torch.stack(pos_imgs, dim=0)
        pos_label = torch.tensor([anchor_label]).repeat(self.pos_transform.num_transforms)
        hard_negs = self.hardneg_dict.get(anchor_label, [])
        hard_neg_imgs = []
        hard_neg_labels = []
        if len(hard_negs) >= self.num_hardnegs:
            hard_neg_idx = torch.randperm(len(hard_negs))[:self.num_hardnegs]
            for i in hard_neg_idx:
                img_i, label_i = hard_negs[i]
                hard_neg_imgs.append(self.hardneg_transform(img_i))
                hard_neg_labels.append(label_i)
        else:
            for i in range(len(hard_negs)):
                img_i, label_i = hard_negs[i]
                hard_neg_imgs.append(self.hardneg_transform(img_i))
                hard_neg_labels.append(label_i)

            num_remaining_hard_negs = self.num_hardnegs - len(hard_negs)
            neg_idx = torch.randperm(len(self.dataset))
            ctr = 0
            for i in neg_idx:
                img_i, label_i = self.dataset[i]
                if label_i == anchor_label:
                    continue
                hard_neg_imgs.append(self.random_transform(img_i))
                hard_neg_labels.append(label_i)
                ctr += 1
                if ctr == num_remaining_hard_negs:
                    break
        hard_neg_imgs = torch.stack(hard_neg_imgs, dim=0)
        hard_neg_labels = torch.tensor(hard_neg_labels)

        random_idx = torch.randperm(len(self.dataset))[:self.num_randoms]
        random_imgs = []
        random_labels = []
        for i in random_idx:
            img_i, label_i = self.dataset[i]
            random_imgs.append(self.random_transform(img_i))
            random_labels.append(label_i)
        random_imgs = torch.stack(random_imgs, dim=0)
        random_labels = torch.tensor(random_labels)

        x = torch.cat([pos_imgs, hard_neg_imgs, random_imgs], dim=0)
        y = torch.cat([pos_label, hard_neg_labels, random_labels], dim=0)

        return x, y
    
def train_val_split(dataset, val_ratio):
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset

def get_misclassified_images(model, loader, pil_transform):
    model.eval()

    misclassified = []

    for inputs, labels in tqdm.tqdm(loader):

        with torch.no_grad():
            logits = model(inputs)
            preds = logits.argmax(dim=1)

            mis_idx = (preds != labels).nonzero(as_tuple=True)[0]

            for i in mis_idx:
                misclassified.append((inputs[i], labels[i], preds[i]))  # (img, true_label, predicted_label)

    misclassified_dict = {}
    for img, true_label, pred_label in misclassified:
        if pred_label.item() not in misclassified_dict:
            misclassified_dict[pred_label.item()] = []
        misclassified_dict[pred_label.item()].append((pil_transform(img), true_label.item()))

    return misclassified_dict

def save_misclassified_images(misclassified_dict, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    for pred_label, images_and_true_labels in misclassified_dict.items():
        folder_name = f"{classes[pred_label]}"
        folder_path = os.path.join(save_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        count = 0
        for img, true_label in images_and_true_labels:
            save_path = os.path.join(folder_path, f"{count}_{classes[true_label]}.png")
            img.save(save_path)
            count += 1
            
        print(f"Saved {count} images to {folder_path}")