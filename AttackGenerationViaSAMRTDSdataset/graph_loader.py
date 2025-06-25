import pandas as pd
import torch
import copy
import numpy as np
#from torch_geometric.utils import get_laplacian
import scipy.sparse as sps


def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping




def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        weight = np.array([complex(elem) for elem in df.values])
        return torch.from_numpy(weight).view(-1, 1).to(self.dtype)


def single2batch_int(edge_index_single, edge_label_single, num_batch):

    edge_index_numpy = copy.deepcopy(edge_index_single.numpy())
    edge_label_numpy = copy.deepcopy(edge_label_single.numpy())
    edge_index_batch = np.zeros((edge_index_numpy.shape[0], edge_index_numpy.shape[1]*num_batch))
    edge_label_batch = np.zeros((edge_label_numpy.shape[0]*num_batch, edge_label_numpy.shape[1]), dtype = 'complex_')
    for i in range(num_batch):
        edge_index_batch[:, edge_index_numpy.shape[1]* i: edge_index_numpy.shape[1]* (i+1)] = edge_index_numpy + (np.max(edge_index_numpy)+1)*i
        edge_label_batch[ edge_label_numpy.shape[0] *i : edge_label_numpy.shape[0] * (i+1),:] = edge_label_numpy 


    return torch.from_numpy(edge_index_batch).to(dtype=torch.long), torch.from_numpy(edge_label_batch).to(dtype=torch.complex64)


def single2batch_phy(Y_sparse: sps._csc.csc_matrix, num_batch: int):
    # 1) csc matrix to coo matrix, to get row, col, data
    coo_mat = Y_sparse.tocoo() 
    # 2) real row/column data, into torch.Tensor
    # edge_index_single size e.g., [2, 421]
    edge_index_single = torch.stack(
        [
            torch.from_numpy(coo_mat.row).long(),
            torch.from_numpy(coo_mat.col).long()
        ],
        dim=0
    )
    # edge_label_single size is e.g., [421, 1]，cplx number
    edge_label_single = torch.from_numpy(coo_mat.data.astype(np.complex64)).view(-1, 1)
    edge_index_batch, edge_label_batch = single2batch_int(
        edge_index_single, edge_label_single, num_batch
    )
    # print('index', torch.max(edge_index_batch))
    return edge_index_batch, edge_label_batch


def single2batch(num_batch, rating_path):
    df= pd.read_csv(rating_path)
    _, user_mapping  = load_node_csv(rating_path, index_col='Bus_F')
    _, movie_mapping = load_node_csv(rating_path, index_col='Bus_T')

    edge_index_single, edge_label_single = load_edge_csv(
        rating_path,
        src_index_col='Bus_F',
        src_mapping=user_mapping,
        dst_index_col='Bus_T',
        dst_mapping=user_mapping,
        encoders={'Weight': IdentityEncoder(dtype=torch.complex64)},
    )
    edge_index_batch,  edge_label_batch = single2batch_int(edge_index_single, edge_label_single, num_batch)
    # print('index', torch.max(edge_index_batch))
    return edge_index_batch, edge_label_batch

def tempo2graph(num_batch, num_tmp):
    Bus_TF=np.linspace(0, num_tmp-2, num_tmp-1, endpoint=True)
    Bus_TT=np.linspace(1, num_tmp-1, num_tmp-1, endpoint=True)

    Bus_TF_bi = np.hstack((Bus_TF, Bus_TT))
    Bus_TT_bi = np.hstack((Bus_TT, Bus_TF))

    Bus_TF_batch = Bus_TF_bi.tolist()
    Bus_TT_batch = Bus_TT_bi.tolist()
    


    # Bus_TF_batch = Bus_TF_list 
    # Bus_TT_batch = Bus_TT_list

    for i in range(num_batch-1):


        Bus_TF_bi = Bus_TF_bi+num_tmp
        Bus_TT_bi = Bus_TT_bi+num_tmp

        Bus_TF_list = Bus_TF_bi.tolist()
        Bus_TT_list = Bus_TT_bi.tolist()

        Bus_TF_batch = Bus_TF_batch + Bus_TF_list 
        Bus_TT_batch = Bus_TT_batch + Bus_TT_list 

    Bus_TF_batch = np.array(Bus_TF_batch).astype(int)
    Bus_TT_batch = np.array(Bus_TT_batch).astype(int)

    edge_index_batch2 = np.vstack((Bus_TF_batch,Bus_TT_batch))

    edge_label_batch2 = np.ones(Bus_TF_batch.shape).astype(int)

    return torch.from_numpy(edge_index_batch2).to(dtype=torch.long), torch.from_numpy(edge_label_batch2)



def single2batch_GT(num_batch, num_tmp):
    edge_index_batch_S, edge_label_batch_S = single2batch(num_batch)
    edge_index_batch_T, edge_label_batch_T = tempo2graph(num_batch, num_tmp)
    return edge_index_batch_S, edge_label_batch_S, edge_index_batch_T, edge_label_batch_T




# edge_index_batch, edge_label_batch = single2batch(2)
# A= get_laplacian(edge_index_batch, edge_label_batch)
# print(A)

# print('edge_index_batch', edge_index_batch.shape)

# print('edge_label_batch', edge_label_batch.shape)

#edge_index_batch_S, edge_label_batch_S, edge_index_batch_T, edge_label_batch_T = single2batch_GT(2, 5)



