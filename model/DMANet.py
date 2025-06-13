import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import math
from timm.models.helpers import named_apply
from timm.models.layers import trunc_normal_tf_
from functools import partial
from model.swinTransformer import SwinTransformer
from torchvision import models as resnet_model


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer



class CAB(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)*x




class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.Conv1 = BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.Conv2 = BasicConv2d(in_channel, out_channel, kernel_size=5, stride=1, padding=2)
        self.Conv3 = BasicConv2d(in_channel, out_channel, kernel_size=7, stride=1, padding=3)

        self.decay_weight2 = nn.Parameter(torch.zeros(1, 1, 1))
        self.decay_weight3 = nn.Parameter(torch.zeros(1, 1, 1))

        self.CAB = CAB(out_channel)
        self.relu = nn.ReLU()
        self.Cat = BasicConv2d(3 * out_channel, out_channel, 1, 1)
        self.res_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                                       nn.BatchNorm2d(out_channel))

        nn.init.trunc_normal_(self.decay_weight2, a=0.5, b=0.75)
        nn.init.trunc_normal_(self.decay_weight3, a=0.25, b=0.5)

    def forward(self, x):
        ideneity = x
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)

        x1_ch = x1
        x2_ch = self.decay_weight2 * x2
        x3_ch = self.decay_weight3 * x3

        x_cat = torch.cat((x1_ch, x2_ch, x3_ch), dim=1)
        x_cat = self.Cat(x_cat)
        x_ca = self.CAB(x_cat)

        output = self.relu(x_ca + self.res_conv(ideneity))

        return output



class MSFM(nn.Module):
    def __init__(self, channel):
        super(MSFM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 9, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)



class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out

class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()
        self.bn_acti = bn_acti
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        if self.bn_acti:
            self.bn_relu = BNPReLU(out_channels)

    def forward(self, input):
        output = self.conv(input)
        if self.bn_acti:
            output = self.bn_relu(output)

        return output

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = Conv(in_channel, out_channel, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)
        self.res_conv =nn.Conv2d(in_channel, out_channel, 1)
        self.res_back_conv = nn.Conv2d(out_channel, in_channel, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        f1 = self.relu(x + self.res_conv(identity))
        f2 = torch.mul(F.sigmoid(self.res_back_conv(f1)) + 1, identity)
        x = self.conv(f2)
        x = self.relu(self.res_conv(f2) + x)
        return x


class DMANET(nn.Module):
    def __init__(self,num_classes=9,dim=320):
        super(DMANET, self).__init__()
        self.num_classes = num_classes

        resnet = resnet_model.resnet34(pretrained = True) # pretrained = True
        self.backbone = SwinTransformer()

        self.mscm1 = MSCM(448, 128)
        self.mscm2 = MSCM(896, 128)
        self.mscm3 = MSCM(1792, 128)

        self.msfm = MSFM(128)

        self.conv3 = nn.Conv2d(in_channels=9, out_channels=1792, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=9, out_channels=896, kernel_size=(1, 1))
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=448, kernel_size=(1, 1))

        self.cab3 = CoordAtt(1792,1792)
        self.cab2 = CoordAtt(896,896)
        self.cab1 = CoordAtt(448,448)

        self.Conv1_1 = ConvLayer(1792, 128)
        self.Conv1_2 = ConvLayer(128, 128)
        self.Conv1_3 = ConvLayer(128, 128)
        self.Conv1_4 = ConvLayer(128, 128)
        self.Conv1_5 = Conv(128, 9, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.Conv2_1 = ConvLayer(896, 128)
        self.Conv2_2 = ConvLayer(128, 128)
        self.Conv2_3 = ConvLayer(128, 128)
        self.Conv2_4 = ConvLayer(128, 128)
        self.Conv2_5 = Conv(128, 9, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.Conv3_1 = ConvLayer(448, 128)
        self.Conv3_2 = ConvLayer(128, 128)
        self.Conv3_3 = ConvLayer(128, 128)
        self.Conv3_4 = ConvLayer(128, 128)
        self.Conv3_5 = Conv(128, 9, 3, 1, 1, dilation=(1, 1), groups=1, bn_acti=True, bias=False)

        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.resnet_layer1 = resnet.layer1
        self.resnet_layer2 = resnet.layer2
        self.resnet_layer3 = resnet.layer3
        self.resnet_layer4 = resnet.layer4

        self.SE_1 = SEBlock(4*dim + 512)
        self.SE_2 = SEBlock(2*dim + 256)
        self.SE_3 = SEBlock(dim + 128)

        self.out = nn.Conv2d(dim, num_classes,kernel_size=1)

    def forward(self, x):
        res = self.first_conv(x)
        res = self.first_bn(res)
        res = self.first_relu(res)

        res1 = self.resnet_layer1(res)
        res2 = self.resnet_layer2(res1)
        res3 = self.resnet_layer3(res2)
        res4 = self.resnet_layer4(res3)

        swin_features = self.backbone(x)
        swin_feature1 = swin_features[0]
        swin_feature2 = swin_features[1]
        swin_feature3 = swin_features[2]

        cat_1 = torch.cat([swin_feature3, res4], dim=1)
        cat_1 = self.SE_1(cat_1)
        cat_2 = torch.cat([swin_feature2, res3], dim=1)
        cat_2 = self.SE_2(cat_2)
        cat_3 = torch.cat([swin_feature1, res2], dim=1)
        cat_3 = self.SE_3(cat_3)

        cat_3_m = self.mscm1(cat_3)
        cat_2_m = self.mscm2(cat_2)
        cat_1_m = self.mscm3(cat_1)
        agg = self.msfm(cat_1_m,cat_2_m,cat_3_m)
        pre_map = F.interpolate(agg, scale_factor=4, mode='bilinear')

        Feature_map_3 = F.interpolate(agg, scale_factor=0.25, mode='bilinear')
        x = -1 * (torch.sigmoid(Feature_map_3)) + 1
        x = self.conv3(x)
        x = self.cab3(x)
        x = x.mul(cat_1)
        x = self.Conv1_1(x)
        x = self.Conv1_2(x)
        x = self.Conv1_3(x)
        x = self.Conv1_4(x)
        conv3_feature = self.Conv1_5(x)
        x = conv3_feature + Feature_map_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')

        Feature_map_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(Feature_map_2)) + 1
        x = self.conv2(x)
        x = self.cab2(x)
        x = x.mul(cat_2)
        x = self.Conv2_1(x)
        x = self.Conv2_2(x)
        x = self.Conv2_3(x)
        x = self.Conv2_4(x)
        conv2_feature = self.Conv2_5(x)
        x = conv2_feature + Feature_map_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')

        Feature_map_1 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1 * (torch.sigmoid(Feature_map_1)) + 1
        x = self.conv1(x)
        x = self.cab1(x)
        x = x.mul(cat_3)
        x = self.Conv3_1(x)
        x = self.Conv3_2(x)
        x = self.Conv3_3(x)
        x = self.Conv3_4(x)
        conv1_feature = self.Conv3_5(x)
        x = conv1_feature + Feature_map_1
        lateral_map_1 = F.interpolate(x, scale_factor=4, mode='bilinear')



        return pre_map , lateral_map_3 , lateral_map_2 , lateral_map_1



if __name__ == "__main__":
    x = torch.rand(2,3,224,224).cuda()
    model = DMANET().cuda()
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)