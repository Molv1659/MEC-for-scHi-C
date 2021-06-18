import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn.functional as nnf
from sklearn.model_selection import train_test_split

from lib.stackedDAE import StackedDAE
from lib.dec import DEC

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from lib.stackedDAE import StackedDAE
from lib.dec import DEC

resize_shape = 150

# 读数据
filename = 'path/to/cell_matrix_scHiCluster_01.npy'
data = np.load(filename, allow_pickle=True)

# 数据处理成 [cell_num, chrom_num, H, W]的格式
cell_matrix = []
for i in range(len(data)):
    chrom_matrix = []
    for matrix in data[i].values():
        matrix_tensor = torch.Tensor(matrix)

        # A + A.T恢复对称
        matrix_tensor = matrix_tensor + torch.transpose(matrix_tensor, 0, 1)

        # resize
        x = matrix_tensor.unsqueeze_(0).unsqueeze_(0)
        x = nnf.interpolate(x, size=(resize_shape, resize_shape), mode='bilinear', align_corners=False).view(1, resize_shape*resize_shape) 

        chrom_matrix.append(x)
    cell_i = torch.cat(chrom_matrix, 0).view(1, 23, resize_shape*resize_shape)
    # print('cell_i',cell_i.shape)
    cell_matrix.append(cell_i)
cell_matrix = torch.cat(cell_matrix, 0)
print('cell_matrix', cell_matrix.shape)

# label转化成数字
labels = np.load('path/to/cell_label.npy')
str2num_dict = {'HeLa':0, 'HAP1':1, 'GM12878':2, 'K562':3}
num_labels = [str2num_dict[i] for i in labels]
num_labels = torch.Tensor(num_labels)

# finetune
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

dec = DEC(input_dim=resize_shape*resize_shape, z_dim=30, n_clusters=4,
    encodeLayer=[300], activation="relu", dropout=0)
print(dec)
sdae_savepath = ("model/sdae_300_30_lr_0001_epoch_30_symmetric_150.pt")
dec.load_model(sdae_savepath)
records = dec.fit(cell_matrix, num_labels, lr=0.003, batch_size=32, num_epochs=300, update_interval=1, tol=-1, kmeans_init=True)

# 存实验训练变化结果，方便画曲线图
with open('sdae_300_30_lr_0001_epoch_30_symmetric_150_finetune_lr_003_epoch_300', 'w') as f:
    f.write(str(records))