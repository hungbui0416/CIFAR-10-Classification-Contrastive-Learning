import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, contrast_mode='scl', temperature=0.1):
        super().__init__()
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, projs, labels):
        device = projs.device
        bs = labels.shape[0]
        v = projs.shape[0] // bs
        projs = F.normalize(projs, dim=-1)

        if self.contrast_mode == 'scl':
            labels = labels.contiguous().view(-1, 1).repeat(v, 1)
        elif self.contrast_mode == 'simclr':
            labels = torch.arange(bs, device=device).view(-1, 1).repeat(v, 1)
        else:
            raise ValueError(f"Invalid contrast mode: {self.contrast_mode}")

        label_mask = torch.eq(labels, labels.T).float().to(device)
        anchor_mask = ~torch.eye(bs * v, dtype=torch.bool, device=device)
        pos_mask = label_mask * anchor_mask

        sims = torch.matmul(projs, projs.T)
        logits = sims / self.temperature

        logit_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logit_max.detach()

        exp_logits = torch.exp(logits_stable) * anchor_mask
        log_prob = logits_stable - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        num_pos = pos_mask.sum(1).clamp(min=self.eps)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss

class HardNegContrastiveLoss(nn.Module):
    def __init__(self, contrast_mode='scl', temperature=0.1):
        super().__init__()
        self.contrast_mode = contrast_mode
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, projs, labels):
        device = labels.device
        hcl_bs, hard_neg_bs = labels.shape
        projs = F.normalize(projs, dim=-1)

        if self.contrast_mode == 'scl':
            labels = labels.contiguous()
            label_mask = (labels == labels[:, 0].view(-1, 1)).float()
            anchor_mask = torch.ones_like(label_mask, device=device)
            anchor_mask[:, 0] = 0
            pos_mask = label_mask * anchor_mask
        elif self.contrast_mode == 'simclr':
            anchor_mask = torch.ones(hcl_bs, hard_neg_bs, device=device)
            anchor_mask[:, 0] = 0
            pos_mask = torch.zeros(hcl_bs, hard_neg_bs, device=device)
            pos_mask[:, 1] = 1.
        else:
            raise ValueError(f"Invalid contrast mode: {self.contrast_mode}")

        anchors = projs[:, 0].unsqueeze(1)
        sims = torch.sum(anchors*projs, dim=-1)
        logits = sims / self.temperature

        logit_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits_stable = logits - logit_max.detach()

        exp_logits = torch.exp(logits_stable) * anchor_mask
        log_prob = logits_stable - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        num_pos = pos_mask.sum(1).clamp(min=self.eps)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / num_pos

        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss