import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
import datetime
import timm
from torch.cuda.amp import GradScaler, autocast
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os

cv2.setNumThreads(0)
from cv2.ximgproc import thinning

warnings.filterwarnings('ignore')

class EnhancedSpatialAttention(nn.Module):
    """Enhanced spatial attention with multi-scale receptive fields."""
    def __init__(self, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(2, 1, k, padding=k//2, bias=False) for k in kernel_sizes
        ])
        self.fusion = nn.Conv2d(len(kernel_sizes), 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)

        multi_scale_features = []
        for conv in self.convs:
            multi_scale_features.append(conv(attention_input))

        fused = self.fusion(torch.cat(multi_scale_features, dim=1))
        return self.sigmoid(fused)

class AdaptiveSEBlock(nn.Module):
    """Enhanced SE block with adaptive pooling and gating."""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        mid_channels = max(in_channels // reduction, 8)
        self.fc_avg = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        self.fc_max = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)

        avg_out = self.fc_avg(avg_y)
        max_out = self.fc_max(max_y)

        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)
        
class SimpleASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleASPP, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
        ])
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        return self.project(torch.cat(res, dim=1))
        
class CrackCapsule(nn.Module):
    def __init__(self, in_channels, capsule_dim=32, num_capsules=32):
        super(CrackCapsule, self).__init__()
        self.capsule_dim, self.num_capsules = capsule_dim, num_capsules
        self.feature_extractor = SimpleASPP(in_channels, num_capsules * capsule_dim)

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x):
        primary_caps = self.feature_extractor(x)
        primary_caps = primary_caps.view(x.size(0), self.num_capsules, self.capsule_dim, x.size(2), x.size(3))
        primary_caps = primary_caps.permute(0, 3, 4, 1, 2).contiguous()
        return self.squash(primary_caps, dim=-1)

class CrackRoutingByAgreement(nn.Module):
    def __init__(self, input_slots, output_caps, capsule_dim, num_routing=3):
        super(CrackRoutingByAgreement, self).__init__()
        self.input_slots, self.output_caps, self.capsule_dim, self.num_routing = input_slots, output_caps, capsule_dim, num_routing
        self.W = nn.Parameter(torch.randn(1, self.input_slots, self.output_caps, self.capsule_dim, self.capsule_dim))

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / (torch.sqrt(squared_norm) + 1e-8)

    def forward(self, x):
        batch_size = x.size(0)
        x_tiled = x.unsqueeze(2).unsqueeze(4)
        W_tiled = self.W.repeat(batch_size, 1, 1, 1, 1)
        u_hat = torch.matmul(W_tiled, x_tiled).squeeze(4)
        b = torch.zeros(batch_size, self.input_slots, self.output_caps, 1).to(x.device)

        for i in range(self.num_routing):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1, keepdim=True)
            v = self.squash(s, dim=-1)
            if i < self.num_routing - 1:
                agreement = (u_hat * v).sum(dim=-1, keepdim=True)
                b = b + agreement
        return v.squeeze(1)

class ImprovedDecoderBlock(nn.Module):
    """Enhanced decoder block with better feature fusion."""
    def __init__(self, in_channels, skip_channels, out_channels, capsule_dim):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_up = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.spatial_attention = EnhancedSpatialAttention()
        self.channel_attention = AdaptiveSEBlock(skip_channels)

        self.capsule_modulator = nn.Sequential(
            nn.Linear(capsule_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2 * (out_channels + skip_channels))
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.se_block = AdaptiveSEBlock(out_channels)
        self.shortcut = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x_up, x_skip, capsule_vector):
        x = self.up_sample(x_up)
        x = self.conv_up(x)

        if x.shape[2:] != x_skip.shape[2:]:
            x = F.interpolate(x, size=x_skip.shape[2:], mode='bilinear', align_corners=False)

        x_skip_channel_att = self.channel_attention(x_skip)
        x_skip_spatial_att = self.spatial_attention(x_skip_channel_att) * x_skip_channel_att

        combined = torch.cat([x, x_skip_spatial_att], dim=1)

        mod = self.capsule_modulator(capsule_vector).view(capsule_vector.size(0), -1, 1, 1)
        scale, bias = torch.chunk(mod, 2, dim=1)
        modulated_features = combined * (scale.sigmoid() * 2) + bias

        main_path = self.fusion_conv(modulated_features)
        main_path = self.se_block(main_path)
        main_path = self.dropout(main_path)

        shortcut_path = self.shortcut(modulated_features)

        return F.relu(main_path + shortcut_path)

class MultiScaleHyperColumn(nn.Module):
    """Enhanced hypercolumn with multi-scale feature aggregation."""
    def __init__(self, decoder_channels):
        super().__init__()
        total_channels = sum(decoder_channels)

        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(total_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(total_channels, 64, 5, padding=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(total_channels, 64, 7, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout2d(0.2),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.attention = EnhancedSpatialAttention()

    def forward(self, d_features, target_size):
        upsampled_features = [
            F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            for f in d_features
        ]
        hyper_features = torch.cat(upsampled_features, dim=1)

        scale_outputs = [processor(hyper_features) for processor in self.scale_processors]
        multi_scale = torch.cat(scale_outputs, dim=1)
        fused = self.fusion(multi_scale)
        attention_map = self.attention(fused)
        return fused * attention_map

class EnhancedCrackCapsuleNetwork(nn.Module):
    """Enhanced version with better feature processing."""
    def __init__(self, num_final_caps=24, capsule_embedding_dim=128, img_size=448):
        super().__init__()

        self.backbone = timm.create_model('efficientnet_b4', features_only=True, pretrained=True, out_indices=(1, 2, 3, 4))
        encoder_channels = self.backbone.feature_info.channels()

        primary_caps_dim, primary_caps_num = 48, 48
        self.crack_capsules = CrackCapsule(encoder_channels[-1], primary_caps_dim, primary_caps_num)

        num_input_slots = (img_size // 32) ** 2 * primary_caps_num
        self.crack_routing = CrackRoutingByAgreement(num_input_slots, num_final_caps, primary_caps_dim, num_routing=4)

        self.capsule_head = nn.Sequential(
            nn.Linear(num_final_caps * primary_caps_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, capsule_embedding_dim)
        )

        self.decoder_channels = [384, 192, 96]
        self.decoder3 = ImprovedDecoderBlock(encoder_channels[3], encoder_channels[2], self.decoder_channels[0], capsule_embedding_dim)
        self.decoder2 = ImprovedDecoderBlock(self.decoder_channels[0], encoder_channels[1], self.decoder_channels[1], capsule_embedding_dim)
        self.decoder1 = ImprovedDecoderBlock(self.decoder_channels[1], encoder_channels[0], self.decoder_channels[2], capsule_embedding_dim)

        self.hypercolumn = MultiScaleHyperColumn(decoder_channels=self.decoder_channels)

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=1)
        )

        self.ds_head3 = nn.Conv2d(self.decoder_channels[0], 1, kernel_size=1)
        self.ds_head2 = nn.Conv2d(self.decoder_channels[1], 1, kernel_size=1)
        self.ds_head1 = nn.Conv2d(self.decoder_channels[2], 1, kernel_size=1)

    def forward(self, x):
        s1, s2, s3, s4 = self.backbone(x)

        primary_caps = self.crack_capsules(s4)
        b, h, w, n_caps, d_caps = primary_caps.shape
        primary_caps_flat = primary_caps.view(b, h * w * n_caps, d_caps)

        if primary_caps_flat.shape[1] != self.crack_routing.input_slots:
            target_slots = self.crack_routing.input_slots
            primary_caps_flat = F.interpolate(
                primary_caps_flat.transpose(1, 2),
                size=target_slots,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)

        routed_capsules = self.crack_routing(primary_caps_flat)
        capsule_embedding = self.capsule_head(routed_capsules.view(b, -1))

        d3 = self.decoder3(s4, s3, capsule_embedding)
        d2 = self.decoder2(d3, s2, capsule_embedding)
        d1 = self.decoder1(d2, s1, capsule_embedding)

        hyper_out = self.hypercolumn([d3, d2, d1], target_size=x.shape[2:])
        output = self.final_conv(hyper_out)

        outputs = {
            'segmentation_logits': output[:, 0:1, :, :],
            'width_map': torch.sigmoid(output[:, 1:2, :, :]),
        }
        return outputs