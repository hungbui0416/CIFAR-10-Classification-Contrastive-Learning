import torch

import tqdm
import wandb

def sl_batch_log(step, loss, optimizer, log_freq):
    if step % log_freq == 0:
        wandb.log({
            "train/loss": loss,
            "train/lr": optimizer.param_groups[0]['lr'],
        })

def sl_epoch_log(epoch, epochs, train_loss, train_acc, val_loss, val_acc):
    print(f"epoch [{epoch:2}/{epochs}] train_loss {train_loss:.4f} train_acc {train_acc:.2f}%")
    print(f"epoch [{epoch:2}/{epochs}] val_loss {val_loss:.4f} val_acc {val_acc:.2f}%")
    
    wandb.log({
        "train/epoch_loss": train_loss,
        "train/epoch_acc": train_acc,
        "val/epoch_loss": val_loss,
        "val/epoch_acc": val_acc,
        "epoch": epoch
    })

def sl_train(epoch, loader, device, model, criterion, optimizer, scheduler, log_freq):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0

    for step, (images, labels) in enumerate(tqdm.tqdm(loader, desc=f"Train epoch {epoch}")):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        sl_batch_log(step, loss.item(), optimizer, log_freq)

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

def sl_validate(epoch, loader, device, model, criterion):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm.tqdm(loader, desc=f"Val epoch {epoch}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    acc = 100. * correct / total
    return avg_loss, acc

class EarlyStopper:
    def __init__(self, model, model_dir, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.model = model
        self.model_dir = model_dir
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            print(f'best_val_loss {val_loss:.4f}, save model!')
            torch.save(self.model.module.state_dict(), self.model_dir)
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def test(loader, device, model, criterion):
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            y_true.append(labels)
            y_pred.append(predicted)

    avg_loss = total_loss / total
    acc = 100. * correct / total

    y_true = torch.cat(y_true, dim=0).to('cpu').numpy()
    y_pred = torch.cat(y_pred, dim=0).to('cpu').numpy()
    
    return avg_loss, acc, y_true, y_pred

def cl_batch_log(step, log_freq, loss, optimizer):
    if step % log_freq == 0:
        wandb.log({
            "train/loss": loss,
            "train/lr": optimizer.param_groups[0]['lr'],
        })


def cl_epoch_log(epoch, num_epochs, train_loss):
    print(f"epoch [{epoch:2}/{num_epochs}] train_loss {train_loss:.4f}")
    
    wandb.log({
        "train/epoch_loss": train_loss,
        "epoch": epoch
    })

def cl_train(epoch, loader, device, model, criterion, optimizer, scheduler, log_freq):
    model.train()
    
    total_loss = 0
    total = 0

    for step, (images, labels) in enumerate(tqdm.tqdm(loader, desc=f"Train epoch {epoch}")):
        images = torch.cat(images, dim=0).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        projs = model(images)
        loss = criterion(projs, labels) 
        optimizer.zero_grad()       
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        cl_batch_log(step, log_freq, loss.item(), optimizer)

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / total
    return avg_loss

def hcl_train(epoch, loader, device, model, criterion, optimizer, scheduler, log_freq):
    model.train()
    
    total_loss = 0
    total = 0

    for step, (images, labels) in enumerate(tqdm.tqdm(loader, desc=f"Train epoch {epoch}")):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        hcl_batch_size, hard_neg_batch_size, C, H, W = images.shape
        images = images.view(-1, C, H, W)
        projs = model(images)
        projs = projs.view(hcl_batch_size, hard_neg_batch_size, -1)
        loss = criterion(projs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * hcl_batch_size
        total += hcl_batch_size

        cl_batch_log(step, log_freq, loss.item(), optimizer)

    if scheduler:
        scheduler.step()

    avg_loss = total_loss / total
    return avg_loss