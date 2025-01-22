import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import norm
import torch
from sklearn import preprocessing
from . import models
import argparse
import configparser
import os
import pandas as pd
from database import get_db
from models import linkidsortorder
import logging
logging.basicConfig(level=logging.INFO)
def get_link_id_sort_order():
    db = next(get_db())
    link_sort_sort_order=db.query(linkidsortorder.matrix_index,linkidsortorder.link_id).all()
    link_sort_sort_order = pd.DataFrame([(int(row[0]), int(row[1])) for row in link_sort_sort_order])
    link_sort_sort_order.columns = ['matrix_index','link_id']
    return link_sort_sort_order

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
    args = parser.parse_args([]) # 스케줄러 실행 시 기본 값 사용

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
def calculation(dataM):
    args, blocks=get_parameters()
    checkpoint_path="./predict/data/Weight.pt"
    dense_matrix=sp.load_npz('./predict/data/adj_matrix.npz')
    adj = sp.csc_matrix(dense_matrix)
    n_vertex = adj.shape[0]
    gso = calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv' or 'OSA':
        gso = calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex).cpu()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    zscore = preprocessing.StandardScaler()  
    x=np.zeros([1,6,24,n_vertex])
    x[0,:,:,:]=dataM.copy()
    x[:,0,:,:]=zscore.fit_transform(x[:,0,:,:].reshape(-1,1)).reshape(1,24, 1876)
    logging.info('Model parameter generation completed.Inference started.') 
    example_input=torch.tensor(x.copy().astype(dtype=np.float32)).cpu()
    y = model(example_input).squeeze(1).numpy()[0, :, :]
    for i in range(0, 3):
        y[i, :] = zscore.inverse_transform(y[i, :].reshape(-1, 1).T)
    y = y.T
    Result = pd.DataFrame(y)
    Result['5 min'] = Result[0]
    Result['10 min'] = Result[1]
    Result['15 min'] = Result[2]
    Result = Result.drop([0, 1, 2], axis=1)
    return Result

def process_data(df_combined,weakday,time_series_sorted):
    indexlist=['tm', 'link_id', 'speed', 'nx', 'ny', 'pty', 'rn1']
    df_combined['holiday']=1.0*weakday
    time_series_sorted['matrix_index'] = range(len(time_series_sorted))
    df_combined['date'] =df_combined['tm'].dt.strftime('%Y%m%d').astype(int)
    df_combined['time'] =df_combined['tm'].dt.strftime('%H%M').astype(int)
    #matrices['M1']['tm'] = (matrices['M1']['tm'].dt.strftime('%Y%m%d%H%M').astype(float).astype(int))
    df_combined['link_id']=(df_combined['link_id'].astype(float).astype(int))
    linkidsortorder=get_link_id_sort_order()
    link_order = linkidsortorder.sort_values('matrix_index')['link_id'].tolist()
    timeorder=time_series_sorted['Time'].tolist()
    base_df = pd.DataFrame({
        'tm': np.repeat(timeorder.copy(), len(link_order.copy())),
        'link_id': np.tile(link_order.copy(), len(timeorder.copy()))
    })
    base_df['date'] = base_df['tm'].dt.strftime('%Y%m%d').astype(int)
    base_df['time'] = base_df['tm'].dt.strftime('%H%M').astype(int)
    base_df['tm'] = base_df['tm'].dt.strftime('%Y%m%d%H%M').astype(int)
    add_ptime_column(base_df)
    add_pdate_column(base_df)
    logging.info("Reference matrix generation completed.")
    finaldata=df_combined
    pivot_speed_p = finaldata.pivot_table(
                index=['tm'],
                columns='link_id',
                values='speed',
                aggfunc='mean',
                fill_value=0
            ).reindex(columns=link_order, fill_value=0).reindex(index=timeorder,columns=link_order, fill_value= np.nan)
    pivot_speed_interpolated = pivot_speed_p.interpolate(method='linear', axis=0)
    pivot_speed = pivot_speed_interpolated.copy()
    if bool(np.isnan(pivot_speed).any().any()):
        nan_count = np.isnan(pivot_speed).sum()

    finaldata=df_combined.copy()
    pivot_holiday = finaldata.pivot_table(
                index=['tm'],
                columns='link_id',
                values='holiday',
                aggfunc='mean',
                fill_value=finaldata['holiday'][0]
    ).reindex(columns=link_order, fill_value=finaldata['holiday'][0]).reindex(index=timeorder,columns=link_order, fill_value=finaldata['holiday'][0])
    finaldata=base_df.copy()
    pivot_date = finaldata.pivot_table(
                index=['tm'],
                columns='link_id',
                values='pdate',
                aggfunc='mean',
                fill_value=finaldata['pdate'][0]
    ).reindex(columns=link_order,fill_value=finaldata['pdate'][0]).reindex(index=timeorder,columns=link_order, fill_value=finaldata['pdate'][0])
    finaldata=base_df.copy()
    pivot_time = finaldata.pivot_table(
        index=['tm'], 
        columns='link_id',
        values='ptime',
        aggfunc='mean',
        fill_value=0
    ).reindex(index=timeorder, fill_value=0).reindex(index=timeorder,columns=link_order, fill_value=0)
    finaldata=df_combined.copy()
    pivot_PTY_p = finaldata.pivot_table(
        index=['tm'],
        columns='link_id',
        values='pty',
        aggfunc='mean',
        fill_value=0
    ).reindex(columns=link_order, fill_value=0).reindex(index=timeorder,columns=link_order, fill_value= np.nan)
    pivot_PTY_interpolated = pivot_PTY_p.interpolate(method='linear', axis=0)
    pivot_PTY = pivot_PTY_interpolated
    if bool(np.isnan(pivot_PTY).any().any()):
        nan_count = np.isnan(pivot_PTY).sum()
        pivot_PTY = pivot_PTY.fillna(0)

    finaldata=df_combined.copy()
    pivot_RN1_p = finaldata.pivot_table(
        index=['tm'],
        columns='link_id',
        values='rn1',
        aggfunc='mean',
        fill_value=0
    ).reindex(columns=link_order, fill_value=0).reindex(index=timeorder,columns=link_order, fill_value= np.nan)
    pivot_RN1_interpolated = pivot_RN1_p.interpolate(method='linear', axis=0)
    pivot_RN1 = pivot_RN1_interpolated
    if bool(np.isnan(pivot_RN1).any().any()):
        nan_count = np.isnan(pivot_RN1).sum()
        pivot_RN1 = pivot_RN1.fillna(0)

    batch_speed_array=np.expand_dims(pivot_speed.to_numpy(dtype='float32'),axis=0)
    batch_holiday_array = np.expand_dims(pivot_holiday.to_numpy(dtype='float32'),axis=0)
    batch_date_array= np.expand_dims(pivot_date.to_numpy(dtype='float32'),axis=0)
    batch_time_array= np.expand_dims(pivot_time.to_numpy(dtype='float32'),axis=0)
    batch_PTY_array= np.expand_dims(pivot_PTY.to_numpy(dtype='float32'),axis=0)
    batch_RN1_array= np.expand_dims(pivot_RN1.to_numpy(dtype='float32'),axis=0)
    feature_array = np.concatenate([batch_speed_array, batch_holiday_array,batch_date_array,batch_time_array,batch_PTY_array,batch_RN1_array], axis=0)
    logging.info("Model input tensor creation completed")
    Result=calculation(feature_array)
    Result['link_id']=linkidsortorder['link_id']
    return Result    

def save_config(config, path):
    # 1) 디렉토리 경로 추출
    dir_path = os.path.dirname(path)

    # 2) 디렉토리가 없는 경우 생성 (중간 디렉토리까지 모두)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # 3) 파일 쓰기
    try:
        with open(path, 'w') as f:
            config.write(f)
        print(f"파일 저장 완료: {path}")
    except Exception as e:
        print("파일 저장 중 오류 발생:", e)

def add_ptime_column(df):
    def hhmm_to_minutes(hhmm):
        hours = hhmm // 100  # HH 부분
        minutes = hhmm % 100  # MM 부분
        time_minutes=hours * 60 + minutes 
        return  np.sin((time_minutes / 1440) * 2 * np.pi)
    df['ptime'] = df['time'].apply(hhmm_to_minutes)
    return df
def add_pdate_column(df):
    def date_to_sine_or_zero(date):
        # 월과 일을 추출
        month = (date // 100) % 100
        day = date % 100
        # 유효한 날짜인지 확인
        if 1 <= month <= 12 and 1 <= day <= 31:
            # 12개월 주기의 사인 값 계산
            return np.sin((month / 12) * 2 * np.pi)
        else:
            return 0  # 유효하지 않으면 0 반환
    # Pdate 열 추가
    df['pdate'] = df['date'].apply(date_to_sine_or_zero)
    return df
