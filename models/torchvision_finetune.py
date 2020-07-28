import torch
import torch.nn as nn
import torchvision

def get_model(arch='resnet50', pretrained=True, n_class=8, dropout=False):

    classifier = nn.ModuleList()
    if dropout:
        classifier.append(nn.Dropout(p=0.2, inplace=False))
    classifier.append(nn.Linear(get_feature_dim(arch), n_class))

    if arch == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Sequential(*classifier)
    elif arch == 'resnet34':
        net = torchvision.models.resnet34(pretrained=pretrained)
        net.fc = nn.Sequential(*classifier)
    elif arch == 'resnet50':
        net = torchvision.models.resnet50(pretrained=pretrained)
        net.fc = nn.Sequential(*classifier)
    elif arch == 'wide_resnet50':
        net = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        net.fc = nn.Sequential(*classifier)
    elif arch == 'resnext50':
        net = torchvision.models.resnext50_32x4d(pretrained=pretrained)
        net.fc = nn.Sequential(*classifier)
    elif arch == 'vgg16':
        net = torchvision.models.vgg16(pretrained=pretrained)
        
        if not dropout:
            net.classifier[2] == nn.Identity()
            net.classifier[5] == nn.Identity()
        net.classifier[6] = nn.Linear(get_feature_dim(arch), n_class)
    elif arch == 'mnasnet':
        net = torchvision.models.mnasnet1_0(pretrained=pretrained)
        net.classifier = nn.Sequential(*classifier)
    elif arch == 'mobilenet':
        net = torchvision.models.mobilenet_v2(pretrained=pretrained)
        net.classifier = nn.Sequential(*classifier)
    else:
        raise NotImplementedError

    return net

def get_feature_dim(arch):
    feature_dict = {
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'wide_resnet50': 2048,
        'resnext50': 2048,
        'vgg16': 4096,
        'mnasnet': 1280,
        'mobilenet': 1280
    }

    if arch not in feature_dict.keys():
        raise NotImplementedError
    
    return feature_dict[arch]