import torch
from torch import nn
from fixkernel import Hybrid

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.con1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.con2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.block=nn.Sequential(
            self.con1,
            self.bn1,
            self.relu,
            self.con2,
            self.bn2,
        )
        self.shortcut=nn.Identity()
        if stride!=1 or in_channels !=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        out=self.relu(
            self.block(x)+self.shortcut(x)
        )
        return out

class ResNet(nn.Module):
    def __init__(self,num_class=176):
        super().__init__()
        self.in_channels=128
        self.hybrid=Hybrid()
        # self.conv1=nn.Conv2d(in_channels=3,out_channels=128,kernel_size=7,stride=2,padding=3,bias=False)
        # self.bn1=nn.BatchNorm2d(128)
        # self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1=self._make_layer(64,2,1)
        self.layer2=self._make_layer(128,2,2)
        self.layer3=self._make_layer(256,2,2)
        self.layer4=self._make_layer(512,2,2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512,num_class)
        # self.fc = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, num_class)
        # )
        self.dropout=nn.Dropout(p=0.3)
        self.net=nn.Sequential(
            self.hybrid,
            # self.conv1,
            # self.bn1,
            # self.relu,
            self.maxpool,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
            nn.Flatten(),
            self.dropout,
            self.fc
        )
    def _make_layer(self,out_channels,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for s in strides:
            layers.append(BasicBlock(self.in_channels,out_channels,s))
            self.in_channels=out_channels

        return nn.Sequential(*layers)

    def forward(self,x):
        out=self.net(x)
        return out





