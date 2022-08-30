import sys
import backbones.net as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):
    def __init__(self, backbone='MyNet', num_classes=2,embed_dim=None):
        super(Network, self).__init__()
        self.backbone_name = backbone
        # if "_" in backbone:
        #     self.backbone = ImageNetmodels.__dict__[backbone](num_classes=num_classes, backbone_fc=False)
        # else:
        #     self.backbone = models.__dict__[backbone](num_classes=num_classes,backbone_fc=False)
        self.backbone = models.__dict__[backbone](num_classes=num_classes,backbone_fc=False) #128 * 1 * 18
        self.dim = self.get_backbone_last_layer_out_channel() #128 * 1 * 18
        if embed_dim:
            self.embeddingLayer =nn.Sequential(
                nn.Linear(self.dim, embed_dim), #128 * 1 * 18 -> 2
                nn.PReLU(),
            )
            self.dim = embed_dim
        self.classifier = nn.Linear(self.dim, num_classes) # 2 -> 2

    def get_backbone_last_layer_out_channel(self):
        if self.backbone_name == "LeNetPlus":
            return 128 * 3 * 3
        if self.backbone_name == "MyNet":
            return 128 * 1 * 18
        # last_layer = list(self.backbone.children())[-1]
        # while (not isinstance(last_layer, nn.Conv2d)) and \
        #         (not isinstance(last_layer, nn.Linear)) and \
        #         (not isinstance(last_layer, nn.BatchNorm2d)):

        #         temp_layer = list(last_layer.children())[-1]
        #         if isinstance(temp_layer, nn.Sequential) and len(list(temp_layer.children()))==0:
        #             temp_layer = list(last_layer.children())[-2]
        #         last_layer = temp_layer
        # if isinstance(last_layer, nn.BatchNorm2d):
        #     return last_layer.num_features
        # elif isinstance(last_layer, nn.Linear):
        #     return last_layer.out_features
        # else:
        #     return last_layer.out_channels

    def forward(self, x):
        feature = self.backbone(x)
        if feature.dim()==4:
            feature = F.adaptive_avg_pool2d(feature,1)
            feature = feature.view(x.size(0), -1)
        # if includes embedding layer.
        feature = self.embeddingLayer(feature) if hasattr(self, 'embeddingLayer') else feature  #128 * 1 * 18 -> 2
        logits = self.classifier(feature) # 2 -> 2
        return feature, logits


def demo():
    # this demo didn't test metaembedding, should works if defined the centroids.
    # x = torch.rand([1,1, 28, 28])
    # net = Network('LeNetPlus',  50,2)
    # feature, logits = net(x)
    # print(feature.shape) #torch.Size([1, 2])
    # print(logits.shape)  #torch.Size([1, 50])

    x = torch.rand([1, 1, 1, 18])
    print(x)
    print(x.size())
    net = Network('MyNet', 2, 2)
    feature, logits = net(x)
    print(feature.shape) #torch.Size([1, 2])
    print(feature)
    print(logits.shape) #torch.Size([1, 2])
    print(logits)

    x = [-1.2611e-03, -5.0107e-03, -5.5262e-04, -4.3407e-03, -5.1200e-05,
           -3.4819e-03, -4.9054e+00,  3.2057e+00,  1.0536e+01,  2.0784e+00,
            1.1622e+01,  3.8205e+00,  1.1497e+02,  3.2958e+01,  7.7813e+01,
            3.1822e+04,  3.2292e+00,  1.3294e+01]
    x = np.array(x)
    x = x.astype(np.float32)
    x = torch.from_numpy(x)
    x = x.reshape(1,1,1,18)
    print(x)
    print(x.size())
    net = Network('MyNet', 2, 2)
    feature, logits = net(x)
    print(feature.shape)
    print(feature)  
    print(logits.shape)
    print(logits)

#demo()