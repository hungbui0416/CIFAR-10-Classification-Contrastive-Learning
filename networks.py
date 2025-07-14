import torch
import torch.nn as nn
from torch import Tensor

from typing import Optional
import math

# CNN
class CNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(64)
        self.layer2 = self.make_layer(128, downsample=True)
        self.layer3 = self.make_layer(256, downsample=True)
        self.layer4 = self.make_layer(512, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, planes: int, downsample: bool = False) -> nn.Sequential:
        layer = [
            nn.Conv2d(self.in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        ]
        if downsample:
            layer.append(nn.Sequential(
                nn.MaxPool2d(2, 2)
            ))
        
        self.in_planes = planes

        return nn.Sequential(*layer)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64)
        self.layer2 = self._make_layer(128, 2)
        self.layer3 = self._make_layer(256, 2)
        self.layer4 = self._make_layer(512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes), 
            )

        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes
        layers.append(BasicBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# ViT
class NewGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class PatchEmbeddings(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_channels: int, emb_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.projection = nn.Conv2d(self.num_channels, self.emb_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x): # (B, C, H, W)
        x = self.projection(x) # (B, emb_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, emb_dim)
        return x


class Embeddings(nn.Module):
    def __init__(self, image_size: int, patch_size: int, num_channels: int, emb_dim: int, dropout: float):
        super().__init__()
        self.patch_embeddings = PatchEmbeddings(image_size, patch_size, num_channels, emb_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # (B, C, H, W)
        x = self.patch_embeddings(x) # (B, num_patches, emb_dim)
        batch_size, _, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, emb_dim)
        x = torch.cat([cls_tokens, x], dim=1) # (B, num_patches + 1, emb_dim)
        x = x + self.position_embeddings # (B, num_patches + 1, emb_dim)
        x = self.dropout(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, emb_dim: int, attention_head_size: int, dropout: float, bias: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.attention_head_size = attention_head_size
        self.query = nn.Linear(emb_dim, attention_head_size, bias=bias)
        self.key = nn.Linear(emb_dim, attention_head_size, bias=bias)
        self.value = nn.Linear(emb_dim, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x): # (B, num_patches + 1, emb_dim)
        query = self.query(x) # (B, num_patches + 1, attention_head_size)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) # (B, num_patches + 1, num_patches + 1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value) # (B, num_patches + 1, attention_head_size)
        return attention_output


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_attention_heads: int, dropout: float, bias: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.emb_dim // self.num_attention_heads
        assert self.attention_head_size * self.num_attention_heads == self.emb_dim, "emb_dim must be divisible by num_attention_heads"
        self.bias = bias
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.emb_dim,
                self.attention_head_size,
                dropout,
                self.bias
            )
            self.heads.append(head)
        self.output_projection = nn.Linear(self.emb_dim, self.emb_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        attention_outputs = [head(x) for head in self.heads] # (B, num_patches + 1, attention_head_size)
        attention_output = torch.cat([attention_output for attention_output in attention_outputs], dim=-1) # (B, num_patches + 1, emb_dim)
        attention_output = self.output_projection(attention_output) # (B, num_patches + 1, emb_dim)
        attention_output = self.output_dropout(attention_output)
        return attention_output 


class FasterMultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, num_attention_heads: int, dropout: float, bias: bool):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = self.emb_dim // self.num_attention_heads
        assert self.attention_head_size * self.num_attention_heads == self.emb_dim, "emb_dim must be divisible by num_attention_heads"
        self.bias = bias
        self.qkv_projection = nn.Linear(self.emb_dim, self.emb_dim * 3, bias=self.bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.emb_dim, self.emb_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        qkv = self.qkv_projection(x) # (B, num_patches + 1, emb_dim * 3)
        query, key, value = torch.chunk(qkv, 3, dim=-1) # (B, num_patches + 1, emb_dim)
        batch_size, sequence_length, _ = query.size()
        query = query.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2) # (B, num_attention_heads, num_patches + 1, attention_head_size)
        key = key.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) # (B, num_attention_heads, num_patches + 1, num_patches + 1)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        attention_output = torch.matmul(attention_probs, value) # (B, num_attention_heads, num_patches + 1, attention_head_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.emb_dim) # (B, num_patches + 1, emb_dim)
        attention_output = self.output_projection(attention_output) # (B, num_patches + 1, emb_dim)
        attention_output = self.output_dropout(attention_output)
        return attention_output


class MLP(nn.Module):
    def __init__(self, emb_dim: int, intermediate_dim: int, dropout: float):
        super().__init__()
        self.dense_1 = nn.Linear(emb_dim, intermediate_dim)
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(intermediate_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        x = self.dense_1(x) # (B, num_patches + 1, intermediate_dim)
        x = self.activation(x)
        x = self.dense_2(x) # (B, num_patches + 1, emb_dim)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, emb_dim: int, num_attention_heads: int, dropout: float, bias: bool, use_faster_attention: bool, intermediate_dim: int):
        super().__init__()
        self.use_faster_attention = use_faster_attention
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(emb_dim, num_attention_heads, dropout, bias)
        else:
            self.attention = MultiHeadAttention(emb_dim, num_attention_heads, dropout, bias)
        self.layernorm_1 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, intermediate_dim, dropout)
        self.layernorm_2 = nn.LayerNorm(emb_dim)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        attention_output = self.attention(self.layernorm_1(x)) # (B, num_patches + 1, emb_dim)
        x = x + attention_output
        mlp_output = self.mlp(self.layernorm_2(x)) # (B, num_patches + 1, emb_dim)
        x = x + mlp_output
        return x


class Encoder(nn.Module):
    def __init__(self, num_blocks: int, emb_dim: int, num_attention_heads: int, dropout: float, bias: bool, use_faster_attention: bool, intermediate_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            block = Block(emb_dim, num_attention_heads, dropout, bias, use_faster_attention, intermediate_dim)
            self.blocks.append(block)

    def forward(self, x): # (B, num_patches + 1, emb_dim)
        for block in self.blocks:   
            x = block(x) # (B, num_patches + 1, emb_dim)
        return x


class ViT(nn.Module):
    def __init__(self, image_size: int = 32, patch_size: int = 4, num_channels: int = 3, emb_dim: int = 64, num_classes: int = 10, num_blocks: int = 4, num_attention_heads: int = 4, emb_dropout: float = 0.0, attn_dropout: float = 0.0, bias: bool = True, use_faster_attention: bool = True, intermediate_dim: int = 4*64, initializer_range: float = 0.02):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_attention_heads = num_attention_heads
        self.emb_dropout = emb_dropout
        self.attn_dropout = attn_dropout
        self.bias = bias
        self.use_faster_attention = use_faster_attention
        self.initializer_range = initializer_range
        self.embedding = Embeddings(image_size, patch_size, num_channels, emb_dim, emb_dropout)
        self.encoder = Encoder(num_blocks, emb_dim, num_attention_heads, attn_dropout, bias, use_faster_attention, intermediate_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.apply(self._init_weights)

    def forward(self, x): # (B, C, H, W)
        embedding_output = self.embedding(x) # (B, num_patches + 1, emb_dim)
        encoder_output = self.encoder(embedding_output) # (B, num_patches + 1, emb_dim)
        logits = self.classifier(encoder_output[:, 0]) # (B, num_classes)
        return logits
    
    def _init_weights(self, module):
        if isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.initializer_range,
            ).to(module.cls_token.dtype)