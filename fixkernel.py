import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def create_gabor_kernel(size, theta, lambd, sigma=2.0, gamma=0.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    # 旋转坐标
    x_theta = x * math.cos(theta) + y * math.sin(theta)
    y_theta = -x * math.sin(theta) + y * math.cos(theta)
    # 高斯包络
    gaussian = torch.exp(
        -(x_theta ** 2 + gamma ** 2 * y_theta ** 2)
        / (2 * sigma ** 2)
    )
    # 余弦波
    wave = torch.cos(2 * math.pi * x_theta / lambd)
    kernel = gaussian * wave
    # 去均值和归一化
    kernel = kernel - kernel.mean()
    kernel = kernel / (torch.linalg.vector_norm(kernel) + 1e-8)
    return kernel

class FixedKernelBranch(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)

        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)

        laplacian_1 = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32)

        laplacian_2 = torch.tensor([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=torch.float32)

        kernels_3x3 = torch.stack([
            sobel_x,
            sobel_y,
            laplacian_1,
            laplacian_2
        ]).unsqueeze(1)
        self.register_buffer("kernels_3x3", kernels_3x3)

        gabor_kernels = []
        thetas = [0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]
        lambdas = [3, 4, 5]
        for theta in thetas:
            for lambd in lambdas:
                kernel = create_gabor_kernel(
                    size=7,
                    theta=theta,
                    lambd=lambd
                )
                gabor_kernels.append(kernel)
        #将kernels_7x7注册为神经网络中的一个固定张量，让它跟随模型移动到GPU，但不会被优化器更新。
        kernels_7x7 = torch.stack(gabor_kernels).unsqueeze(1)
        self.register_buffer("kernels_7x7", kernels_7x7)

        # 16 → 64 通道
        self.expand = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gray = (0.299 * x[:, 0:1] +0.587 * x[:, 1:2] +0.114 * x[:, 2:3])
        features_3x3 = F.conv2d(
            gray,
            self.kernels_3x3,
            stride=2,
            padding=1
        )

        features_7x7 = F.conv2d(
            gray,
            self.kernels_7x7,
            stride=2,
            padding=3
        )

        features = torch.cat(
            [features_3x3, features_7x7],
            dim=1
        )

        output = self.expand(features)
        return output

class LearnableBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()

        self.learnable_branch = LearnableBranch()
        self.fixed_branch = FixedKernelBranch()
        self.fusion = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.se = SEAttention(
            channels=128,
            reduction=16
        )

    def forward(self, x):
        feature_learnable = self.learnable_branch(x)
        feature_fixed = self.fixed_branch(x)
        output = torch.cat([feature_learnable, feature_fixed],dim=1)
        output = self.fusion(output)
        output = self.se(output)
        return output


