import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as nnf

import numpy as np
from sklearn.model_selection import train_test_split

from lib.stackedDAE import StackedDAE
from lib.dec import DEC

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
        x = nnf.interpolate(x, size=(150, 150), mode='bilinear', align_corners=False).view(1, 150*150) 
        chrom_matrix.append(x)
    cell_i = torch.cat(chrom_matrix, 0).view(1, 20, 150*150)
    cell_matrix.append(cell_i)
cell_matrix = torch.cat(cell_matrix, 0)
print('cell_matrix', cell_matrix.shape)

# 读label
labels = np.load('path/to/cell_label_3.npy')
labels_num = np.load('path/to/cell_label_num.npy')

# finetune
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

dec = DEC(input_dim=150*150, z_dim=30, n_clusters=3,
    encodeLayer=[300], activation="relu", dropout=0)
print(dec)
sdae_savepath = ("model/sdae_300_30_lr_00002_epoch_100_Adam_symmetric_scHiCluster_data.pt")
dec.load_model(sdae_savepath)
records = dec.fit(cell_matrix, torch.from_numpy(labels_num), lr=0.01, batch_size=32, num_epochs=200, update_interval=1, tol=-1, kmeans_init=True)

# 存实验训练变化结果，方便画曲线图
save_file = "sdae_300_100_30_lr_001_epoch_2_symmetric_finetune_01_200_scHiCluster_data.txt"
with open(save_file, 'w') as f:
    f.write(str(records))