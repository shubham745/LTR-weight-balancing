from __future__ import absolute_import, division, print_function
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
from collections import OrderedDict
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import numpy as np
import os, math
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class TMDLayer(nn.Module):
    def __init__(
        self,
        in_features = 100,
        L_latent = 16,
        epsilon = 0.25,
        base=False
    ):
        super().__init__()
        
        # what should be the dimension here? should this be user input?

        self.pi_list = nn.Sequential(nn.Linear(L_latent, in_features), 
                                                    nn.ReLU(),
                                                    nn.Linear(in_features, 1),
                                                    nn.Sigmoid())
        self.dt = nn.Parameter(torch.FloatTensor([0.1]))

        self.epsilon = epsilon
        self.proj_list = nn.Sequential(nn.Linear(in_features, L_latent))
        self.base = base
    

    def TMD_map(self, x):
        # input x if of size [B, N, d]

        #         print(x.shape)
        
        n,c,h,w = x.shape
        x = x.view(1,n,-1)
        
        x = self.proj_list(x)
        # L = construct from pe

        i_minus_j = x.unsqueeze(2) - x.unsqueeze(1)
        K_epsilon = torch.exp(-1 / (4 * self.epsilon) * (i_minus_j ** 2).sum(dim=3))

        ### construct TMD
        q_epsilon_tilde = K_epsilon.sum(dim=2)
        D_epsilon_tilde = torch.diag_embed(self.pi_list(x).squeeze(2) / q_epsilon_tilde)
        K_tilde = K_epsilon.bmm(D_epsilon_tilde)
        D_tilde = torch.diag_embed(K_tilde.sum(dim=2) +
                                   1e-5 * torch.ones(K_tilde.shape[0], K_tilde.shape[1]).to(x.device))
        L = 1 / self.epsilon * (torch.inverse(D_tilde).bmm(K_tilde)) - torch.eye(K_tilde.shape[1]).to(
            x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        return L

        
    def forward(self, x, f):
        L = self.TMD_map(x).squeeze(0)
        
        if self.base:
            x = x.view(x.size(0), -1)
        target = f(x)
        
        shape_val = target.shape
        
#         print(target.shape, L.shape, self.TMD_map(x).shape, target.view(shape_val[0],-1).shape)
        
        target = (target + self.dt*torch.matmul(L, target.view(shape_val[0],-1)).view(shape_val))

        return target
    

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers=18, isPretrained=False, isGrayscale=False, embDimension=128, poolSize=4, fc_tmd_layer=False):
        super(ResnetEncoder, self).__init__()
        self.path_to_model = '/tmp/models'
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.isGrayscale = isGrayscale
        self.isPretrained = isPretrained
        self.embDimension = embDimension
        self.poolSize = poolSize
        self.featListName = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
        
        resnets = {
            18: models.resnet18, 
            34: models.resnet34,
            50: models.resnet50, 
            101: models.resnet101,
            152: models.resnet152}
        
        resnets_pretrained_path = {
            18: 'resnet18-5c106cde.pth', 
            34: 'resnet34.pth',
            50: 'resnet50-19c8e357.pth',
            101: 'resnet101.pth',
            152: 'resnet152.pth'}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(
                num_layers))

        self.encoder = resnets[num_layers]()

        # check hyperparameter tuning
        # diff latent dimension
#         self.tmd_layer1  = TMDLayer(
#             in_features = 65536,
#             L_latent = 16,  
#             epsilon = 0.25
#         )
#         self.tmd_layer2  = TMDLayer(
#             in_features = 65536,
#             L_latent = 16,  
#             epsilon = 0.25
#         )
        
#         self.tmd_layer3  = TMDLayer(
#             in_features = 65536,
#             L_latent = 16,  
#             epsilon = 0.25
#         )
        self.tmd_layer4  = TMDLayer(
            in_features = 16384,
            L_latent = 32,  
            epsilon = 0.25
        )
    
        self.fc_tmd_layer = fc_tmd_layer
        if self.fc_tmd_layer:
            self.tmd_layer  = TMDLayer(
                in_features = self.num_ch_enc[-1],
                L_latent = 16,  
                epsilon = 0.25,
                base=True
            )

        
        if self.isPretrained:
            print("using pretrained model")
            self.encoder.load_state_dict(
                torch.load(os.path.join(self.path_to_model, resnets_pretrained_path[num_layers])))
            
        if self.isGrayscale:
            self.encoder.conv1 = nn.Conv2d(
                1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        if num_layers > 34:
            self.num_ch_enc[1:] = 2048
        else:
            self.num_ch_enc[1:] = 512
                    
        if self.embDimension>0:
            self.encoder.fc =  nn.Linear(self.num_ch_enc[-1], self.embDimension)
            
        
            

    def forward(self, input_image):
        self.features = []
        
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        
        x = self.encoder.layer1(x)
        #x = self.tmd_layer2(x, self.encoder.layer1)
        self.features.append(x)
        
       
        #x = self.tmd_layer2(x, self.encoder.layer2)
        x = self.encoder.layer2(x)
        self.features.append(x)
        
        #x = self.tmd_layer3(x, self.encoder.layer3) 
        x = self.encoder.layer3(x)
        self.features.append(x)
        
        x = self.tmd_layer4(x, self.encoder.layer4)
        self.features.append(x)
        
        x = F.avg_pool2d(x, self.poolSize)
        

        if self.fc_tmd_layer:
            x = self.tmd_layer(x, self.encoder.fc)
        else:
            x = self.encoder.fc(x)

        return x
    