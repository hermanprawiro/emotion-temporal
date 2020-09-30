import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class Network(nn.Module):
    def __init__(self, backbone='resnet50', pooling='avg', pretrained=True, n_class=10):
        super().__init__()

        if backbone == 'resnet34':
            self.net = torchvision.models.resnet34(pretrained=pretrained)
            self.conv_dim = 512
        elif backbone == 'resnet50':
            self.net = torchvision.models.resnet50(pretrained=pretrained)
            self.conv_dim = 2048
        else:
            self.net = torchvision.models.resnet18(pretrained=pretrained)
            self.conv_dim = 512

        self.modify_output_channels(n_class)

    def forward(self, x, average_logits=True):
        # x is a 5D tensor (batch, channel, frame, height, width)
        # N, C, T, H, W = x.size()
        # x = x.permute(0, 2, 1, 3, 4)
        # x = x.reshape(N * T, C, H, W)

        logits = self.net(x)
        # logits = logits.view(N, T, -1)
        if average_logits:
            logits = logits.mean(1)

        return logits

    def modify_input_channels(self, new_input_channel):
        modules = list(self.net.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        first_conv = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]
        params = [x.clone() for x in first_conv.parameters()]
        has_bias = (len(params) == 2)
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (new_input_channel,) + kernel_size[2:]

        new_kernel = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        new_conv = nn.Conv2d(new_input_channel, first_conv.out_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, bias=has_bias)
        new_conv.weight.data = new_kernel
        if has_bias:
            new_conv.bias.data = params[1].data

        layer_name = list(container.state_dict().keys())[0][:-7] # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)

    def modify_output_channels(self, new_output_channel):
        modules = list(self.net.modules())
        fc = modules[-1]
        container = modules[0]
        has_bias = len(list(fc.parameters())) == 2
        
        new_fc = nn.Linear(fc.in_features, new_output_channel, bias=has_bias)
        
        layer_name = 'fc'
        # replace the fully connected layer
        setattr(container, layer_name, new_fc)
