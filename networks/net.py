# @ FileName: net.py
# @ Author: Alexis
# @ Time: 20-11-28 下午9:17
from torch.utils import model_zoo
from torchvision.models import resnet
# from networks.net_base import *
# from net_base import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torchvision
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg19
# from ViT import ViT
from torch import nn
import torch
from timm.loss import SoftTargetCrossEntropy
from timm.models import swin_small_patch4_window7_224, vit_base_patch16_224, efficientnet_b5, tnt_b_patch16_224, tnt_s_patch16_224
import copy

NUM_BLOCKS = {
    18: [2, 2, 2, 2],
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3]
}

class VGG19(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(VGG19, self).__init__()
        self.in_channels = in_channels
        component = list(vgg19(pretrained=True).children())
        component0 = list(component[0].children())
        component2 = list(component[2].children())
        # 修改网络结构
        component0[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.features = nn.Sequential(*component0, component[1])
        in_features = component2[-1].in_features
        self.classifier = nn.Sequential(*component2[:-1], nn.Linear(in_features, num_classes))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature_layer(self):
        return self.features[-1][-1]

class Res18(resnet.ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(Res18, self).__init__(resnet.BasicBlock, NUM_BLOCKS[18])
        self.in_channels = in_channels
        component = list(resnet18(pretrained=True).children())
        # 修改网络结构
        component[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*(component[:-1]))
        in_features = component[-1].in_features
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature_layer(self):
        return self.features[-2][-1]

class Res50(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(Res50, self).__init__()
        self.in_channels = in_channels
        component = list(resnet50(pretrained=True).children())
        # 修改网络结构
        component[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*(component[:-1]))
        in_features = component[-1].in_features
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature_layer(self):
        return self.features[-2][-1]

class Res101(resnet.ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(Res101, self).__init__(resnet.Bottleneck, NUM_BLOCKS[101])

        component = list(resnet101(pretrained=True).children())
        # 修改网络结构
        component[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = nn.Sequential(*(component[:-1]))
        in_features = component[-1].in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def get_feature_layer(self):
        return self.features[-2][-1]

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class Res152(resnet.ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        super(Res152, self).__init__(resnet.Bottleneck, NUM_BLOCKS[152])
        self.expansion = resnet.Bottleneck.expansion
        self.in_channels = in_channels
        self.load_state_dict(model_zoo.load_url(resnet.model_urls['resnet152']))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * self.expansion, num_classes)
        init_params(self.fc)
        if in_channels != 3 :
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    def forward(self, x):
        if self.in_channels == x.shape[0]:
            # 普通的前向传播
            return self._forward_impl(x)
        elif x.shape[0] % self.in_channels == 0:
            # 多张图片使用相同的特征提取器，做特征融合，此时的 self.in_channels 应为3
            result_list = []
            num_img = x.shape[0] / self.in_channels
            for i in range(num_img):
                result_list.append(self._forward_impl(x[i*self.in_channels:(i+1)*self.in_channels]))
            return torch.cat(result_list,dim=0)

class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()

        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(),  # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(),  # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )

        # load pre_train model
        pre_model = pretrainedmodels.inceptionv4(num_classes=1000,
                                                 pretrained='imagenet')
        self.last_linear = pre_model.last_linear
        self.load_state_dict(pre_model.state_dict())

        self.last_linear = nn.Linear(1536, num_classes)
        init_params(self.last_linear)

    def logits(self, features):
        # Allows image of any size to be processed
        if isinstance(features.shape[2], int):
            adaptive_avg_pool_width = features.shape[2]
        else:
            adaptive_avg_pool_width = features.shape[2].item()
        x = F.avg_pool2d(features, kernel_size=adaptive_avg_pool_width)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

class InceptionResNetV2(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Modules
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b()
        self.repeat = self._make_layer(Block35, scale=0.17, blocks=10)
        self.mixed_6a = Mixed_6a()
        self.repeat_1 = self._make_layer(Block17, scale=0.10, blocks=20)
        self.mixed_7a = Mixed_7a()
        self.repeat_2 = self._make_layer(Block8, scale=0.20, blocks=9)
        self.block8 = Block8(noReLU=True)
        self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
        self.avgpool_1a = nn.AvgPool2d(8, count_include_pad=False)

        # load pre_train model
        pre_model = pretrainedmodels.inceptionresnetv2(num_classes=1000,
                                                       pretrained='imagenet')
        self.last_linear = pre_model.last_linear
        self.load_state_dict(pre_model.state_dict())

        self.last_linear = nn.Linear(1536, num_classes)
        init_params(self.last_linear)

    def _make_layer(self, block, scale, blocks):
        layers = []
        for i in range(blocks):
            layers.append(block(scale=scale))
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

class PnasNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(PnasNet, self).__init__()
        self.features = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
        self.features.last_linear = nn.Linear(4320, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x

class Se_resnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(Se_resnet, self).__init__()
        self.in_channels = in_channels
        self.features = pretrainedmodels.se_resnet50(num_classes=1000, pretrained='imagenet')
        self.features.last_linear = nn.Linear(2048, num_classes)
        if in_channels != 3:
            self.features.layer0 = nn.Sequential(
                nn.Conv2d(in_channels,64,kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
            )

    def forward(self, x):
        # if self.in_channels == x.shape[0]:
        #     # 普通的前向传播
        #     return self.features(x)
        # elif x.shape[0] % self.in_channels == 0:
        #     # 多张图片使用相同的特征提取器，做特征融合，此时的 self.in_channels 应为3
        #     result_list = []
        #     num_img = x.shape[0] / self.in_channels
        #     for i in range(num_img):
        #         result_list.append(self.features(x[i*self.in_channels:(i+1)*self.in_channels]))
        #     return torch.cat(result_list,dim=0)
        return self.features(x)

    def get_feature_layer(self):
        return self.features.layer4[-1]

class Densnet201(nn.Module):
    def __init__(self, num_classes=1000):
        super(Densnet201, self).__init__()
        self.features = torchvision.models.densenet201(pretrained=True)
        self.features.classifier = nn.Linear(1920, num_classes)

    def forward(self, x):
        x = self.features(x)
        return x
        
class MultiBranch(nn.Module):
    def __init__(self, model_list, num_classes, is_fix=True, is_concat=True):
        super(MultiBranch, self).__init__()
        self.subnetwork_num = len(model_list)
        print('subnet branches : {}'.format(self.subnetwork_num))
        self.features = model_list
        self.is_fix = is_fix
        self.is_concat = is_concat
        in_features = 0
                    
        in_features = 512

        self.weights = nn.Parameter(torch.ones((self.subnetwork_num,1)))
        self.features = nn.ModuleList(self.features)
        if is_concat:
            self.classifier = nn.Linear(in_features*self.subnetwork_num, num_classes)
        else:
            self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # assert x.shape[1] % self.subnetwork_num == 0, "picture channels do not match!"
        out = []
        for i in range(self.subnetwork_num):
            temp = x[:,i,:,:].unsqueeze(1)
            if self.is_fix:
                self.features[i].eval()
            temp = self.features[i](temp)   
            temp = temp.flatten(1)
            temp = temp.unsqueeze(1)    # b, 1, feature_dim
            out.append(temp)
        
        if self.is_concat:
            out = torch.cat(out, 2)
            out = torch.squeeze(out,1)
        else:
            out = torch.cat(out,1)
            out = (out * self.weights).sum(1)
        # print('feature shape: {}'.format(out.shape))
        return self.classifier(out)

    def get_feature_layer(self):
        return self.features[-2][-1]

class PatchEmbed_group(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, groups=in_chans)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=4,  num_classes=2):
        super(ViT, self).__init__()        
        vit = vit_base_patch16_224(pretrained=False, in_chans=in_channels, num_classes=num_classes)
        # 修改网络结构
        vit.patch_embed = PatchEmbed_group(in_chans=in_channels)
        in_features = vit.head.in_features
        vit.head = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)

        self.features = vit
        print('Loaded pretrained vision transformer.')
        
    def forward(self, x):
        return self.features(x)

# class Swin_Transformer(nn.Module):
#     def __init__(self, in_channels=3, num_classes=10):
#         super(Swin_Transformer, self).__init__()

#         component = list(resnet101(pretrained=True).children())
#         # 修改网络结构
#         component[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.features = nn.Sequential(*(component[:-1]))
#         in_features = component[-1].in_features
#         self.classifier = nn.Linear(in_features, num_classes)

#     def get_feature_layer(self):
#         return self.features[-2][-1]

#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, 1)
#         return self.classifier(x)

def get_net(model_type, num_classes, in_channels=9, **kwargs):
    if model_type == 'res18':
        return Res18(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'res50':
        return Res50(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'res101':
        return Res101(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'res152':
        return Res152(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'Inc-v4':
        return InceptionV4(num_classes=num_classes)
    elif model_type == 'IncRes-v2':
        return InceptionResNetV2(num_classes=num_classes)
    elif model_type == 'pnasnet':
        return PnasNet(num_classes=num_classes)
    elif model_type == 'se_resnet':
        return Se_resnet(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'densenet':
        return Densnet201(num_classes=num_classes)
    elif model_type == 'efficientnet':
        return efficientnet_b5(pretrained=True, in_chans=in_channels, num_classes=num_classes)
    elif model_type == 'vit':
        return ViT(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'multi_branch':
        if not 'model_list' in kwargs:
            raise ValueError('model feature extractor list should not be none!')
        return MultiBranch(model_list=kwargs['model_list'], num_classes=num_classes, is_fix=kwargs['is_fix'], is_concat=kwargs['is_concat'])
    elif model_type == 'vgg19':
        return VGG19(in_channels=in_channels, num_classes=num_classes)
    elif model_type == 'swin_transformer':
        return swin_small_patch4_window7_224(pretrained=True, in_chans=in_channels, num_classes=num_classes)
    elif model_type == 'tnt_b':
        return tnt_b_patch16_224(pretrained=True, in_chans=in_channels, num_classes=num_classes)
    elif model_type == 'tnt_s':
        return tnt_s_patch16_224(pretrained=True, in_chans=in_channels, num_classes=num_classes)
    else:
        raise ValueError('No model: {}'.format(model_type))

# print(Res50(6,2).features[-2][-1])
#print(Res101(6,2))
# Res101(6,2)
# print(Efficientnet(6,2))
# v = Vision_Transformer().features
# torch.save(v.state_dict(), './vit_weight.pth')
# print(Vision_Transformer().features)

# print(vgg19())

# print(VGG19(9,2))