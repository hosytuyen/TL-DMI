# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
import models.evolve as evolve 
import utils
import models.vgg as vgg
import models.resnet as resnet
import json


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MCNN2(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN2, self).__init__()
        self.feat_dim = 12800
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return feature,out

class MCNN4(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN4, self).__init__()
        self.feat_dim = 128
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return feature,out

class MCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MCNN, self).__init__()
        self.feat_dim = 256
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 5, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return feature,out

class SCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SCNN, self).__init__()
        self.feat_dim = 512
        self.num_classes = num_classes
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out = self.fc_layer(feature)
        return feature,out

class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        res = self.fc2(x)
        return [x, res]


class VGG16_xray8(nn.Module):
    def __init__(self, num_classes=7):
        super(VGG16_xray8, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature = model.features
        self.feat_dim = 2048
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        self.model = model
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        
        return feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)

        return res

class VGG16(nn.Module):
    def __init__(self, n_classes):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG16_TL(nn.Module):
    def __init__(self, n_classes, net, freeze_layers=0):
        super(VGG16_TL, self).__init__()


        if freeze_layers == 0:
            for param in net.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            for i, child in enumerate(net.feature.children()):
                if i <= freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        self.feature = net.feature
        self.feat_dim = 512 * 2 * 2
        self.n_classes = n_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.bn(feature)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class VGG16_vib(nn.Module):
    def __init__(self, n_classes):
        super(VGG16_vib, self).__init__()
        model = torchvision.models.vgg16_bn(pretrained=True)
        self.feature = model.features
        self.feat_dim = 512 * 2 * 2
        self.k = self.feat_dim // 2
        self.n_classes = n_classes
        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Linear(self.k, self.n_classes)
            
    def forward(self, x, mode="train"):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return [feature, out, mu, std]
    
    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
       
        return out

class VGG19(nn.Module):
    def __init__(self, num_of_classes):
        super(VGG19, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])

        self.feat_dim = 512 * 2 * 2
        self.num_of_classes = num_of_classes
        self.fc_layer =  nn.Linear(self.feat_dim, self.num_of_classes)
    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return  feature, out

class VGG19_xray8(nn.Module):
    def __init__(self, num_classes):
        super(VGG19_xray8, self).__init__()
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature = model.features
        self.feat_dim = 2048
        self.num_classes = num_classes
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.bn.bias.requires_grad_(False)  # no shift
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
        self.model = model

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        # print(feature.shape)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return  feature, out


class EfficientNet_b0(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b0, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b0(pretrained=False)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b1(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b1, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b1(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1280
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_b2(nn.Module):
    def __init__(self, n_classes):
        super(EfficientNet_b2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_b2(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1408
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_s2(nn.Module):
    def __init__(self, n_classes,dataset='celeba'):
        super(EfficientNet_v2_s2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_m2(nn.Module):
    def __init__(self, n_classes,dataset='celeba'):
        super(EfficientNet_v2_m2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)

        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_l2(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_l2, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-1])
        self.n_classes = n_classes
        self.feat_dim = 1028
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_s(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_s, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)

        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_m(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_m, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)

        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class EfficientNet_v2_l(nn.Module):
    def __init__(self, n_classes, dataset='celeba'):
        super(EfficientNet_v2_l, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class efficientNet_v2_l_xray(nn.Module):
    def __init__(self, n_classes):
        super(efficientNet_v2_l_xray, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_l(pretrained=True)
        model.features[0][0] =nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.feature = model.features
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class efficientnet_v2_m_xray(nn.Module):
    def __init__(self, n_classes):
        super(efficientnet_v2_m_xray, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_m(pretrained=True)
        
        model.features[0][0] =nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.feature = model.features
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out

class efficientnet_v2_s_xray(nn.Module):
    def __init__(self, n_classes):
        super(efficientnet_v2_s_xray, self).__init__()
        model = torchvision.models.efficientnet.efficientnet_v2_s(pretrained=True)
        model.features[0][0] =nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        self.feature = model.features
        self.n_classes = n_classes
        self.feat_dim = 5120
        self.fc_layer = nn.Linear(self.feat_dim, self.n_classes)
            
    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        return  feature,res

    def predict(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        res = self.fc_layer(feature)
        out = F.softmax(res, dim=1)

        return feature,out


class ResNet18(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet18, self).__init__()
        model = torchvision.models.resnet18(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.feat_dim = 2048 * 1 * 1
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return feature, out

class ResNet34(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet34, self).__init__()
        model = torchvision.models.resnet34(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.feat_dim = 2048 * 1 * 1
        self.num_of_classes = num_of_classes
        
        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return  feature, out

class ResNet34_xray8(nn.Module):
    def __init__(self, num_of_classes):
        super(ResNet34_xray8, self).__init__()
        model = torchvision.models.resnet34(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.feat_dim = 2048 * 1 * 1
        self.num_of_classes = num_of_classes

        self.fc_layer = nn.Linear(self.feat_dim, self.num_of_classes)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return  feature, out


class Mobilenet_v3_small(nn.Module):
    def __init__(self, num_of_classes):
        super(Mobilenet_v3_small, self).__init__()
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.feat_dim = 2304
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_of_classes),)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return   feature,out

class Mobilenet_v2(nn.Module):
    def __init__(self, num_of_classes):
        super(Mobilenet_v2, self).__init__()
        model = torchvision.models.mobilenet_v2(pretrained=True)
        self.feature = nn.Sequential(*list(model.children())[:-2])
        self.feat_dim = 12288
        self.num_of_classes = num_of_classes
        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_of_classes),)

    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self, x):
        feature = self.feature(x)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)
        return   feature,out

class EvolveFace(nn.Module):
    def __init__(self, num_of_classes, IR152):
        super(EvolveFace, self).__init__()
        if IR152:
            model = evolve.IR_152_64((64,64))
        else:
            model = evolve.IR_50_64((64,64))
        self.model = model
        self.feat_dim = 512
        self.num_classes = num_of_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(p=0.15),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))

        self.fc_layer = nn.Sequential(
            nn.Linear(self.feat_dim, self.num_classes),)


    def classifier(self, x):
        out = self.fc_layer(x)
        __, iden = torch.max(out, dim = 1)
        iden = iden.view(-1, 1)
        return out, iden

    def forward(self,x):
        feature = self.model(x)
        feature = self.output_layer(feature)
        feature = feature.view(feature.size(0), -1)
        out, iden = self.classifier(feature)

        return  out

class FaceNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceNet, self).__init__()
        self.feature = evolve.IR_50_112((112, 112))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)

    def predict(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return out
            
    def forward(self, x):
        feat = self.feature(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return [feat, out]

class FaceNet64(nn.Module):
    def __init__(self, num_classes = 1000, freeze_layers=0):
        super(FaceNet64, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))

        if freeze_layers > 0:
            for param in self.feature.input_layer.parameters():
                param.requries_grad = False

            for i, child in enumerate(self.feature.body.children()):
                if i <= freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False

        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)
        return feat, out


class IR152(nn.Module):
    def __init__(self, num_classes=1000, freeze_layers=0):
        super(IR152, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))

        if freeze_layers > 0:   
            for param in self.feature.input_layer.parameters():
                param.requries_grad = False

            for i, child in enumerate(self.feature.body.children()):
                if i <= freeze_layers:
                    for param in child.parameters():
                        param.requires_grad = False


        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.fc_layer = nn.Linear(self.feat_dim, self.num_classes)
            
    def forward(self, x):
        feat = self.feature(x)
        feat = self.output_layer(feat)
        feat = feat.view(feat.size(0), -1)
        out = self.fc_layer(feat)
        return feat, out

class IR152_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR152_vib, self).__init__()
        self.feature = evolve.IR_152_64((64, 64))
        self.feat_dim = 512
        self.k = self.feat_dim // 2
        self.n_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu#, st


class IR50(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.num_classes = num_classes
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim = 1))

    def forward(self, x):
        feature = self.output_layer(self.feature(x))
        feature = feature.view(feature.size(0), -1)
        statis = self.st_layer(feature)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feature, out, iden, mu, std

class IR50_vib(nn.Module):
    def __init__(self, num_classes=1000):
        super(IR50_vib, self).__init__()
        self.feature = evolve.IR_50_64((64, 64))
        self.feat_dim = 512
        self.n_classes = num_classes
        self.k = self.feat_dim // 2
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                        nn.Dropout(),
                                        Flatten(),
                                        nn.Linear(512 * 4 * 4, 512),
                                        nn.BatchNorm1d(512))  

        self.st_layer = nn.Linear(self.feat_dim, self.k * 2)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.k, self.n_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        feat = self.output_layer(self.feature(x))
        feat = feat.view(feat.size(0), -1)
        statis = self.st_layer(feat)
        mu, std = statis[:, :self.k], statis[:, self.k:]
        
        std = F.softplus(std-5, beta=1)
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        res = mu + std * eps
        out = self.fc_layer(res)
        __, iden = torch.max(out, dim=1)
        iden = iden.view(-1, 1)

        return feat, out, iden, mu, std

def load_TF(args, model_name, n_classes):
    if args[model_name]["public_dataset"] == "pubfig":
        nclass_pub = 83 
    elif args[model_name]["public_dataset"] == "facescrub":
        nclass_pub = 530
    elif args[model_name]["public_dataset"] == "IN1K":
        nclass_pub = 1000
    else:
        print("Public dataset has not supported yet")
        exit()

    if model_name == "VGG16":
        if args[model_name]["public_dataset"] != "IN1K":
            net = VGG16(nclass_pub)
            net = torch.nn.DataParallel(net).cuda()
            ckp_T = torch.load(args[model_name]["resume"])        
            net.load_state_dict(ckp_T['state_dict'], strict=True)
            net.eval()
            net = net.module.to('cpu')
        else:
            net = VGG16(nclass_pub)

    net = VGG16_TL(n_classes, net, freeze_layers=args[model_name]["freeze_layers"])

    return net


def get_classifier(model_name, mode, n_classes, resume_path, args):
    if model_name == "VGG16":
        if args[model_name]["public_dataset"] == "":
            if mode == "reg": 
                net = VGG16(n_classes)
            elif mode == "vib":
                net = VGG16_vib(n_classes)
        else:
            net = load_TF(args, model_name, n_classes)
	
    elif model_name == "FaceNet":
        net = FaceNet(n_classes)

    elif model_name == "FaceNet_all":
        net = FaceNet(202599)
		
    elif model_name == "FaceNet64":
        freeze_layers = args["FaceNet64"]["freeze_layers"]
        net = FaceNet64(n_classes, freeze_layers=freeze_layers)

    elif model_name == "IR50":
        if mode == "reg":
            net = IR50(n_classes)
        elif mode == "vib":
            net = IR50_vib(n_classes)
        
    elif model_name == "IR152":
        freeze_layers = args["IR152"]["freeze_layers"]
        if mode == "reg":
            net = IR152(n_classes, freeze_layers=freeze_layers)
        else:
            net = IR152_vib(n_classes)
    elif model_name == "VGG-R":
        net = vgg.vggr(args[model_name]["type"], n_classes)
    elif model_name == "Resnet-R":
        plain = args["Resnet-R"]["plain"]
        plain = json.loads(plain.lower())
        net = resnet._resnetR(layers=args[model_name]["type"], num_classes=n_classes, plain=plain)
    else:
        print("Model name Error")
        exit()

    if model_name in ['FaceNet', 'FaceNet_all', 'FaceNet_64', 'IR50', 'IR152']:
        if resume_path is not "":
            print("Resume")
            utils.load_state_dict(net.feature, torch.load(resume_path))
        else:
            print("No Resume")    
        

    return net
