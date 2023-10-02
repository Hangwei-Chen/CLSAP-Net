import torch as torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import math
import torch.utils.model_zoo as model_zoo
from function import normal
from function import calc_mean_std

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)     # 64,112,112
        n1,c1,h1,w1= x1.shape

        x2_1 = self.maxpool(x1)
        x2 = self.layer1(x2_1)   # 256,56,56
        n2,c2,h2,w2= x2.shape
        # the same effect as lda operation in the paper, but save much more memory

        x3 = self.layer2(x2)  # 512,28,28
        n3,c3,h3,w3= x3.shape
        x4 = self.layer3(x3)  # 1024,14,14
        n4,c4,h4,w4= x4.shape
        x5 = self.layer4(x4)    # 2048,7,7
        n5,c5,h5,w5= x5.shape
        # out = x



        return x1, x2, x3, x4
def resnet50_backbone(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        save_model = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    else:
        model.apply(weights_init_xavier)
    return model
def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class CPE_Net(nn.Module):

    def __init__(self):
        super(CPE_Net, self).__init__()
        self.content_fusion=content_fusion(64,256,512)
        self.res_content = resnet50_backbone(pretrained=True)
        in_dim = 48
        self.Content_SA1 = CAM(in_dim)
        self.Content_SA2 = CAM(in_dim)

        self.content_Q_pool = nn.Sequential(
            nn.Conv2d(96, 48, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 16, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(7,stride=7),
        )

        self.content_W_pool = nn.Sequential(
            nn.Conv2d(96, 48, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 16, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(7, stride=7),
        )
        self.Q_L1 = nn.Linear(1024, 512)
        self.Q_L2 = nn.Linear(512, 1)
        self.W_L1 = nn.Linear(1024,256)

    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, C_img, O_img):

        res_out_C1,res_out_C2,res_out_C3,_ = self.res_content(C_img)
        res_out_OC1,res_out_OC2,res_out_OC3,_, = self.res_content(O_img)
        res_out_C= self.content_fusion(res_out_C1,res_out_C2,res_out_C3)
        res_out_OC = self.content_fusion(res_out_OC1, res_out_OC2, res_out_OC3)

        O_content_feature = self.Content_SA1(res_out_OC)
        C_content_feature = self.Content_SA2(res_out_C)
        #_______________________________________________________________________________
        Dif_cf = C_content_feature - O_content_feature
        Fusion_cf = torch.cat((Dif_cf, O_content_feature), 1)
       #__________________________________________________________________________________

        Q_content_feature= self.content_Q_pool(Fusion_cf).view(Fusion_cf.size(0), -1)
        Q_content= self.Q_L1(Q_content_feature)
        Q_content= self.Q_L2(Q_content).squeeze()

        Q_content_w_vector=self.content_W_pool(Fusion_cf)
        Q_content_w_vector = Q_content_w_vector.view(-1, self.num_flat_features(Q_content_w_vector))
        content_w_vec = self.W_L1(Q_content_w_vector).view(-1, 256, 1, 1)

        return Q_content,content_w_vec

'''CAM module'''
class CAM(nn.Module):
    def __init__(self, in_dim):
        super(CAM, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content_feat):
        B, C, H, W = content_feat.size()
        F_Fc_norm = self.f(normal(content_feat)).view(B, -1, H * W).permute(0, 2, 1)

        B, C, H, W = content_feat.size()
        G_Fs_norm = self.g(normal(content_feat)).view(B, -1, H * W)

        energy = torch.bmm(F_Fc_norm, G_Fs_norm)
        attention = self.softmax(energy)

        H_Fs = self.h(content_feat).view(B, -1, H * W)
        out = torch.bmm(H_Fs, attention.permute(0, 2, 1))
        B, C, H, W = content_feat.size()
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        out += content_feat

        return out



class content_fusion(nn.Module):
    def __init__(self, channel1,channel2,channel3):
        super(content_fusion, self).__init__()
        self.upx2 = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.upx4 = torch.nn.UpsamplingNearest2d(scale_factor=4)
        self.demo1 = nn.Sequential( #64.112.112
            nn.Conv2d(channel1, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.demo2 = nn.Sequential(  #256.56.56
            nn.Conv2d(channel2, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),

        )
        self.demo3 = nn.Sequential(  #512.28.28
            nn.Conv2d(channel3, 32, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),

        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 48, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),

        )

    def forward(self, x1,x2,x3):
        out1=self.demo1(x1)
        out2 = self.demo2(x2)
        out3 = self.upx2(self.demo3(x3))
        out = self.conv1(torch.cat((out1,out2,out3),1))
        return out
