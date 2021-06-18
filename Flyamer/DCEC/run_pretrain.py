import torch
import os
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn.functional as nnf
from sklearn.model_selection import train_test_split

from lib.CAE import CAE_BN_5

# 选GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 读数据
filename = 'path/to/cell_matrix_scHiCluster_01.npy'
data = np.load(filename, allow_pickle=True)

# 数据处理为  [cell_num * chrom_num, H, W]的格式
chrom_matrix = []
for i in range(len(data)):
    for matrix in data[i].values():
        matrix_tensor = torch.Tensor(matrix)

        # A + A.T恢复对称
        matrix_tensor = matrix_tensor + torch.transpose(matrix_tensor, 0, 1)

        # resize
        x = matrix_tensor.unsqueeze_(0).unsqueeze_(0)
        x = nnf.interpolate(x, size=(150, 150), mode='bilinear', align_corners=False) 

        chrom_matrix.append(x)
chrom_num = len(chrom_matrix)
chrom_matrix = torch.cat(chrom_matrix, 0)

# 读label
labels = np.load('path/to/cell_label_num.npy')

# 数据划分
train_data, val_data = train_test_split(chrom_matrix, test_size=0.2, shuffle=True)

# dataset
class chrom_dataset(Dataset):
    def __init__(self, data=data, label=None):
        self.data = data

    def __getitem__(self, index):
        chrom_emb = self.data[index]
        return chrom_emb, chrom_emb

    def __len__(self):
        return len(self.data)

train_set = chrom_dataset(data=train_data)
val_set = chrom_dataset(data=val_data)

# dataloader
batch_size = 32
train_loader = torch.utils.data.DataLoader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=True)
val_loader = torch.utils.data.DataLoader(
                dataset=val_set,
                batch_size=batch_size,
                shuffle=False)


# pretrain CAE
cae = CAE_BN_5(input_shape=[150,150,1], embedding_dim=30, num_clusters=3, filters=[32, 32, 64, 64, 128])
print(cae)
cae.fit(chrom_matrix, labels, train_loader, val_loader, lr=0.001, num_epochs=100, loss_type="mse")

