###################### feature extractor ###########################
import torch
from torch import nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):

    expansion = 2 #4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# define the PyTorch MLP model
#
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim:int = 256):
        # network parameters:
        block = Bottleneck
        layers = [2, 1, 1]
        zero_init_residual=True
        groups=1
        width_per_group=64
        replace_stride_with_dilation=None
        norm_layer=None

        # inherit the superclass properties/methods
        #
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # define the model
        #
        ################## ped_pos net model: ###################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2,2), padding=(0, 0)),
            nn.BatchNorm2d(256)
        )
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(1,1), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=(1, 1), stride=(4,4), padding=(0, 0)),
            nn.BatchNorm2d(512)
        )
        self.relu3 = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_fc = nn.Sequential(
            nn.Linear(256 * block.expansion + 2, features_dim),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0) 
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)           

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, ped_pos, scan, goal):
        ped_in = ped_pos.reshape(-1,2,80,80)
        scan_in = scan.reshape(-1,1,80,80)
        fusion_in = torch.cat((scan_in, ped_in), dim=1)
        
        x = self.conv1(fusion_in)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        identity3 = self.downsample3(x)

        x = self.layer1(x)

        identity2 = self.downsample2(x)

        x = self.layer2(x)

        x = self.conv2_2(x)
        x += identity2
        x = self.relu2(x)

        x = self.layer3(x)

        x = self.conv3_2(x)
        x += identity3
        x = self.relu3(x)

        x = self.avgpool(x)
        fusion_out = torch.flatten(x, 1)

        goal_in = goal.reshape(-1,2)
        goal_out = torch.flatten(goal_in, 1)
        fc_in = torch.cat((fusion_out, goal_out), dim=1)
        x = self.linear_fc(fc_in)  

        return x

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        ped_pos = observations[:, :12800]
        scan = observations[:, 12800:19200]
        goal = observations[:, 19200:]
        return self._forward_impl(ped_pos, scan, goal)