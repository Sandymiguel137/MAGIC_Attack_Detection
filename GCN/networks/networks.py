import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import NoisyLinear
from networks.network_bodies import SimpleBody, AtariBody
from networks.cheb_conv_wt import  ChebConv
from graph_loader import single2batch
from torch_geometric.utils import get_laplacian
from cplxmodule import cplx
from cplxmodule.nn.modules import CplxConv1d
from cplxmodule.nn.modules import CplxLinear
from model.layers import ChebGraphConv
from model.utility import calc_gso, calc_chebynet_gso, cnv_sparse_mat_to_coo_tensor
from cplxmodule.nn import CplxToCplx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CplxReLU = CplxToCplx[torch.nn.ReLU]


class ActorCriticGCN_CNN(nn.Module):
    def __init__(self, input_shape, num_actions):  # [82, 10], T = 10
        super(ActorCriticGCN_CNN, self).__init__()
        # init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
        #              lambda x: nn.init.constant_(x, 0),
        #              nn.init.calculate_gain('relu'))

        self.Channel_T = 10
        self.Channel_S = 10
        self.conv1 = CplxConv1d(in_channels=input_shape[1], out_channels=self.Channel_T, kernel_size=1)
        K_order = 3
        n_feat = input_shape[1]
        enable_bias = True
        self.conv2 = ChebGraphConv(K_order, n_feat, self.Channel_S, enable_bias)   # Size([8 * 16, 6]) # input_shape[1] is the feature, 16 is the channel 
        # self.conv3 = ChebConv(self.Channel_S, input_shape[1], K=4)
        self.fc1 = CplxLinear(input_shape[1] * input_shape[0], 512)  ## missing edge_index edge_weight

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512*2, 1))   # 32, 1  ---> 2, 1

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.actor_linear = init_(nn.Linear(512*2, num_actions)) # 32, 11  ---> 2, 11

        self.train()


    def forward(self, data):
        x, edge_index, edge_weight  = data.x, data.edge_index, data.edge_attr
        batch_or_agent, nodes_num,  T_num= x.shape # [Batch or agent, bus, time]
        # print('x.dtype', x.dtype)
        # print('%%', nodes_num)
        gso_type = "sym_norm_lap"
        GSO =calc_gso(edge_index, edge_weight, nodes_num*batch_or_agent, gso_type)
        # GSO = calc_chebynet_gso(GSO)
        GSO = cnv_sparse_mat_to_coo_tensor(GSO, device)

        x = x.permute(0, 2, 1)  # [Batch or agent, time, bus]
        # print('test x!', x)
        x = self.conv1(x)  # [batch, Channel_T, bus]
        x = CplxReLU()(x)  # modrelu(input, threshold=0.5)
        x = x.permute(0, 2, 1) # [batch, bus, Channel_T]
        x = x.reshape(-1, self.Channel_T)  # [batch * bus, Channel_T]
        # x_im = torch.zeros(x.size(0), x.size(1))
        # z = torch.complex(x, x_im)
        # print('x_shape', x.shape)
        x = self.conv2(x, GSO)# [batch * bus, Tnum]
        x = CplxReLU()(x)  # modrelu(input, threshold=0.5)

        x = x.view(-1, nodes_num, T_num) # [batch, bus, Tnum]

        x = x.view(x.size(0), -1)  # [batch, bus*Tnum]

        x = self.fc1(x) # [batch, 512]
        x = CplxReLU()(x)  # modrelu(input, threshold=0.5)
        # print('x.shape', x.shape)
        
        x_cat = torch.cat((x.real, x.imag), 1)

        value = self.critic_linear(x_cat)
        logits = self.actor_linear(x_cat)

        return logits, value




    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module



