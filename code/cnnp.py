# 论文中原始的CNN网络结构

# construct the proposed CNNP
import torch.nn as nn
from torch import cat

class CNNP(nn.Module): # CNN预测器
    def __init__(self):
        super(CNNP, self).__init__()
        channel = 32
        layers = [nn.Conv2d(1, channel, 3, 1, 1)]
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        self.conv1_0 = nn.Sequential(*layers)

        layers = [nn.Conv2d(1, channel, 5, 1, 2)]
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        self.conv1_1 = nn.Sequential(*layers)

        layers = [nn.Conv2d(1, channel, 7, 1, 3)]
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(channel, channel, 3, 1, 1))
        self.conv1_2 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv3 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv4 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv5 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv6 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv7 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv8 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv9 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv10 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv11 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv12 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv13 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv14 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv15 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv16 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True)]
        self.conv17 = nn.Sequential(*layers)

        layers = [nn.Conv2d(channel, 1, 3, 1, 1)]
        self.conv18 = nn.Sequential(*layers)

    def forward(self, images): # 前向
        out1_0 = self.conv1_0(images)
        out1_1 = self.conv1_1(images)
        out1_2 = self.conv1_2(images)
        out2 = out1_0 + out1_1 + out1_2
        out3 = self.conv3(out2)
        out4 = self.conv4(out3+out2)
        out5 = self.conv5(out4+out2)
        out6 = self.conv6(out5+out2)
        out7 = self.conv7(out6+out2)
        out8 = self.conv8(out7+out2)
        out9 = self.conv9(out8+out2)
        out10 = self.conv10(out9+out2)
        out11 = self.conv11(out10+out2)
        out12 = self.conv12(out11+out2)
        out13 = self.conv13(out12+out2)
        out14 = self.conv14(out13+out2)
        out15 = self.conv15(out14+out2)
        out16 = self.conv16(out15+out2)
        out17 = self.conv17(out16+out2)
        out18 = self.conv18(out17+out2)
        return out18