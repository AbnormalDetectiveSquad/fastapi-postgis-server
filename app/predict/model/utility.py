import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from sklearn import preprocessing
from . import models
import argparse
import configparser
import os

def calc_gso(dir_adj, gso_type):
    n_vertex = dir_adj.shape[0]

    if sp.issparse(dir_adj) == False:
        dir_adj = sp.csc_matrix(dir_adj)
    elif dir_adj.format != 'csc':
        dir_adj = dir_adj.tocsc()

    id = sp.identity(n_vertex, format='csc')

    # Symmetrizing an adjacency matrix
    adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
    #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
    if gso_type == 'sym_renorm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'sym_renorm_lap' or gso_type == 'rw_renorm_lap':
        adj = adj + id
    
    if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
        or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
        row_sum = adj.sum(axis=1).A1
        row_sum_inv_sqrt = np.power(row_sum, -0.5)
        row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
        deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
        # A_{sym} = D^{-0.5} * A * D^{-0.5}
        sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

        if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            sym_norm_lap = id - sym_norm_adj
            gso = sym_norm_lap
        else:
            gso = sym_norm_adj

    elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
        or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
        row_sum = np.sum(adj, axis=1).A1
        row_sum_inv = np.power(row_sum, -1)
        row_sum_inv[np.isinf(row_sum_inv)] = 0.
        deg_inv = np.diag(row_sum_inv)
        # A_{rw} = D^{-1} * A
        rw_norm_adj = deg_inv.dot(adj)

        if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            rw_norm_lap = id - rw_norm_adj
            gso = rw_norm_lap
        else:
            gso = rw_norm_adj

    else:
        raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_chebynet_gso(gso):
    if sp.issparse(gso) == False:
        gso = sp.csc_matrix(gso)
    elif gso.format != 'csc':
        gso = gso.tocsc()

    id = sp.identity(gso.shape[0], format='csc')
    # If you encounter a NotImplementedError, please update your scipy version to 1.10.1 or later.
    eigval_max = norm(gso, 2)

    # If the gso is symmetric or random walk normalized Laplacian,
    # then the maximum eigenvalue is smaller than or equals to 2.
    if eigval_max >= 2:
        gso = gso - id
    else:
        gso = 2 * gso / eigval_max - id

    return gso

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='enable CUDA, default as True')
    parser.add_argument('--seed', type=int, default=42, help='set the random seed for stabilizing experiment results')
    parser.add_argument('--dataset', type=str, default='seoul', choices=['metr-la', 'pems-bay', 'pemsd7-m','seoul'])
    parser.add_argument('--n_his', type=int, default=24)#타임 스텝 1시간이면 12개
    parser.add_argument('--n_pred', type=int, default=3, help='the number of time interval for predcition, default as 3')
    parser.add_argument('--time_intvl', type=int, default=5)
    parser.add_argument('--Kt', type=int, default=3)# Temporal Kernel Size
    parser.add_argument('--stblock_num', type=int, default=5)
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--Ks', type=int, default=3, choices=[3, 2])
    parser.add_argument('--graph_conv_type', type=str, default='OSA', choices=['cheb_graph_conv', 'graph_conv','OSA'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap', choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True, help='default as True')
    parser.add_argument('--droprate', type=float, default=0.11934253064493432)
    parser.add_argument('--lr', type=float, default=0.0005500530762294727, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_decay_rate', type=float, default=0.055042254587113923, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.8734346556732533)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--k_threshold', type=float, default=430.6227347591174, help='adjacency_matrix threshold parameter menual setting')
    parser.add_argument('--complexity', type=int, default=18, help='number of bottleneck chnnal | in paper value is 16')
    parser.add_argument('--features', type=int, default='6', help='number of features')
    parser.add_argument('--fname', type=str, default=f'run_num_2', help='name')
    parser.add_argument('--mode', type=str, default='train', help='test or train')
    parser.add_argument('--HotEncoding', type=str, default="On", help='On or Off')
    parser.add_argument('--Continue', type=str, default="False", help='True or False')
    args = parser.parse_args()     
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    blocks.append([args.features])
    n=args.complexity
    for l in range(args.stblock_num):
        blocks.append([int(n*4), int(n), int(n*4)])
    if Ko == 0:
        blocks.append([int(n*8)])
    elif Ko > 0:
        blocks.append([int(n*8), int(n*8)])
    blocks.append([1])
    return args, blocks

@torch.no_grad()
def TestScript():
    args, blocks=get_parameters()
    dense_matrix=sp.load_npz('./data/adj_matrix.npz')
    adj = sp.csc_matrix(dense_matrix)
    n_vertex = adj.shape[0]
    gso = calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv' or 'OSA':
        gso = calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso)
    model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex).cpu()
    model.eval() 
    zscore = preprocessing.StandardScaler()  
    x=np.zeros([1,6,24,n_vertex]) 
    x[:,0,:,:]=zscore.fit_transform(x[:,0,:,:].reshape(-1,1)).reshape(1,24, 1876)
    example_input=torch.tensor(x.copy().astype(dtype=np.float32)).cpu()
    y = model(example_input).squeeze(1).numpy()[0, :, :]
    for i in range(0, 3):
        y[i, :] = zscore.inverse_transform(y[i, :].reshape(-1, 1).T)
    return y



class Datareader():
    def __init__(self, path='./model/config.ini',option=None,dim=None):
        self.config = configparser.ConfigParser()
        self.path=path
        self.dim=dim
        if not os.path.exists(self.path):
            self.write_config()
        self.config.read(self.path)
        self.process_data = self.create_dynamic_function()
        if option == 'test':
            self.testarrays = self.generate_test_arrays()
        else:
            self.testarrays = None
        
    def write_config(self, path=None, dim=2): 
        if not path:
            path=self.path
        if self.dim:
            dim=self.dim
        config = configparser.ConfigParser()
        if dim == 2:
            config['arrays'] = {
                'number': '3',
                'format': 'dataframe',
                'channal0': 'velocity',
                'channal1': 'PTY',
                'channal2': 'RN1',
            }
            config['array1'] = {
                'Demension': '2',
                'axis1': '24',    
                'type1' : 'TIme',   
                'axis2': '1876',
                'type2' : 'link'   
            }
            config['array2'] = {
                'Demension': '2',
                'axis1': '24',    
                'type1' : 'TIme',   
                'axis2': '1876',
                'type2' : 'link'  
            }
            config['array3'] = {
                'Demension': '2',
                'axis1': '24',    
                'type1' : 'TIme',   
                'axis2': '1876',
                'type2' : 'link'  
            }
        elif dim == 3:
            config['arrays'] = {
                'number': '1',
                'format': 'dataframe',
                'channal0': 'velocity',
                'channal1': 'PTY',
                'channal2': 'RN1',
            }
            config['array1'] = {
                'Demension': '3',
                'axis1': '3',    
                'type1' : 'channel', 
                'axis2': '24',    
                'type2' : 'TIme',   
                'axis3': '1876',
                'type3' : 'link'
            }
        with open(path, 'w') as f:
            config.write(f)
    def generate_test_arrays(self):
        n_arrays = int(self.config['arrays']['number'])
        arrays = []
        for i in range(n_arrays):
            array_section = f'array{i+1}'
            dim = int(self.config[array_section]['Demension'])
            shape = [int(self.config[array_section][f'axis{j}']) for j in range(1, dim + 1)]
            arrays.append(np.random.rand(*shape))
        return arrays    
    def create_dynamic_function(self, config=None):
        if not config:
            config = self.config
    # arrays 섹션에서 채널 순서 파악
        channel_order = []
        i = 0
        while f'channal{i}' in config['arrays']:
            channel_order.append(config['arrays'][f'channal{i}'].lower())  # 소문자로 통일
            i += 1
        def process_data(*args):
            if len(args) != len(channel_order):
                raise ValueError(f"Expected {len(channel_order)} input arrays, but got {len(args)}")

            final_array = np.zeros((1, 6, 24, 1876))
            for i, channel_name in enumerate(channel_order):
                array_section = f'array{i+1}'
                
                # 채널 이름으로 처리 방식 결정
                if channel_name in ['holiday', 'time', 'seasons']:
                    # 모든 링크에 동일한 값을 가지는 채널들
                    data = args[i].reshape(-1, 1)  # (24, 1) 또는 (1, 1)
                    final_array[0, i, :, :] = np.broadcast_to(data, (24, 1876))
                else:
                    # 링크별로 다른 값을 가지는 채널들
                    final_array[0, i] = args[i].reshape(24, 1876)
        
            return final_array
        return process_data
            
