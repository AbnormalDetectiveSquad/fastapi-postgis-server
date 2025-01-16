import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from model import models, utility
import scipy.sparse as sp
from datetime import datetime
from zoneinfo import ZoneInfo

def get_parameters():
    parser = argparse.ArgumentParser(description='STGCN')
    parser.add_argument('--enable_cuda', type=bool, default=False, help='enable CUDA, default as True')
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
    parser.add_argument('--droprate', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--weight_decay_rate', type=float, default=0.00000, help='weight decay (L2 penalty)')
    parser.add_argument('--epochs', type=int, default=1000, help='epochs, default as 1000')
    parser.add_argument('--opt', type=str, default='adamw', choices=['adamw', 'nadamw', 'lion'], help='optimizer, default as nadamw')
    parser.add_argument('--step_size', type=int, default=18)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--k_threshold', type=float, default=460.0, help='adjacency_matrix threshold parameter menual setting')
    parser.add_argument('--complexity', type=int, default=16, help='number of bottleneck chnnal | in paper value is 16')
    parser.add_argument('--fname', type=str, default='K460_16base_S250samp_seq_lr0.0001', help='name')
    parser.add_argument('--mode', type=str, default='test', help='test or train')
    parser.add_argument('--HotEncoding', type=str, default="On", help='On or Off')
    args = parser.parse_args()
    device = 'cpu'
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    blocks = []
    if args.HotEncoding == 'On':
        blocks.append([2])
    else:
        blocks.append([1])
    n=args.complexity
    for l in range(args.stblock_num):
        blocks.append([int(n*4), int(n), int(n*4)])
    if Ko == 0:
        blocks.append([int(n*8)])
    elif Ko > 0:
        blocks.append([int(n*8), int(n*8)])
    blocks.append([1])
    return args, device, blocks

def Sort_and_shaping(inputd,idsort,weakdayinform=1):
    if inputd.shape==(1876,25):
        inputd['Link_ID'] = inputd['Link_ID'].astype(str).str.strip()
        idsort['Link_ID'] = idsort['Link_ID'].astype(str).str.strip()
        inputd_indexed = inputd.set_index('Link_ID')
        sorted_inputd = inputd_indexed.loc[idsort['Link_ID']].reset_index()
        data_without_id = sorted_inputd.drop(columns='Link_ID')
        float_array = data_without_id.to_numpy(dtype=float).T
        weakday_array = np.ones([float_array.shape[0],float_array.shape[1]]) * weakdayinform
        output=np.stack([float_array, weakday_array], axis=0).astype(dtype=np.float32)
    elif not (weakdayinform >= 0) and not (weakdayinform <= 1):
        raise ValueError(f'ERROR: The inputd shape is not correct.')
    else:
        raise ValueError(f'ERROR: The inputd shape is not correct.')
    return output

def calculattion_data(input,weakday):        
    
    args, device, blocks = get_parameters()
    ID_sort=pd.read_csv('./model/ID_sort.csv') 
    x=Sort_and_shaping(InPut,ID_sort)                                 
    zscore = preprocessing.StandardScaler()                           
    x[0,:,:] = zscore.fit_transform(x[0,:,:])
    x=torch.tensor(x).unsqueeze(0)

    dense_matrix=sp.load_npz('./model/adj_matrix.npz')
    adj = sp.csc_matrix(dense_matrix)
    n_vertex = adj.shape[0]
    gso = utility.calc_gso(adj, args.gso_type)
    if args.graph_conv_type == 'cheb_graph_conv' or 'OSA':
        gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray()
    gso = gso.astype(dtype=np.float32)
    args.gso = torch.from_numpy(gso)
    with torch.no_grad():
        model = models.STGCNChebGraphConv_OSA(args, blocks, n_vertex)
        model.load_state_dict(torch.load("./model/gangnamgu_with_weakday.pt",map_location=torch.device('cpu')))
        model.eval()
        y = model(x).squeeze(1).numpy()[0,:,:]
    
    for i in range(0,3):
        y[i,:]=zscore.inverse_transform(y[i,:].reshape(-1,1).T)
    
    time_intervals = ["5 min","10 min", "15 min"]
    result = pd.DataFrame(y.T, columns=time_intervals)
    result.insert(0, 'Link_ID', ID_sort['Link_ID'].values)
    return result


async def predict_traffic(db: Session):
    # 최근 2시간 데이터 조회
    # TODO 교통 데이터 조회
    recent_data = ~~~
    # TODO 기상 데이터 조회
    ~~~

    # 요일 정보 계산
    today = datetime.now(tz=ZoneInfo("Asia/Seoul"))
    weekday = today.weekday()
    if weekday == 0:
        weakday = 0
    elif weekday == 4:
        weakday = 0.5
    elif weekday in [5,6]:
        weakday = 1
    else:
        weakday = 0.1

    result = calculattion_data(recent_data, weakday)

    # 예측 결과 저장 (created_at:UTC, 예측 결과:result)
    result['created_at'] = datetime.now()  # UTC 시간
    result = result.rename(columns=
        {
            'Link_ID': 'link_id',
            '5 min': 'prediction_5min',
            '10 min': 'prediction_10min',
            '15 min': 'prediction_15min'
        }
    )

    result.to_sql('traffic_predictions', db.bind, if_exists='append', index=False)
    await db.commit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    InPut=pd.read_csv('./model/Sample.csv')# 입력 데이터 샘플 입니다. 시간순으로 column이 0부터 23 까지 되어있고 Link_ID 
                                 #column에 서울시 강남구의 Link_ID가 들어가 있어야 합니다. 순서는 자동 정렬 됩니다.
    weakday=1 #월요일=0 금요일=0.5 토,일=1, 나머지 요일=0.1

    out=calculattion_data(InPut,weakday)# 출력입니다. Link_ID, 5min, 10min, 15 min column이 있습니다.
    print(out.columns)
    print(out)
    
    