import torch
import os
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn.functional as nnf

from lib.DCEC import DCEC_BN_5

# 选GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 存实验训练变化结果，方便画曲线图
save_file = "pretrain_10_records_output_lr_0001_epoch_30_dcec_loss.txt"
print(save_file)

# 读数据
filename =  'path/to/cell_matrix_scHiCluster_01.npy'
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
        x = nnf.interpolate(x,
                            size=(150, 150),
                            mode='bilinear',
                            align_corners=False).view(1, 150, 150)

        chrom_matrix.append(x)
    cell_i = torch.cat(chrom_matrix, 0).view(1, 20, 150, 150)
    cell_matrix.append(cell_i)
cell_matrix = torch.cat(cell_matrix, 0)
print("cell_matrix:", cell_matrix.shape)

# 读label
num_labels =  np.load('/path/to/cell_label_num.npy')
num_labels = torch.from_numpy(num_labels)

# 加载预训练模型
dcec = DCEC_BN_5(input_shape=[150, 150, 1],
                    embedding_dim=30,
                    num_clusters=3,
                    filters=[32, 32, 64, 64, 128])
dcec.load_model("model/scHiCluster_data_cae_bn_5_lr_001_epoch_10_mse_symmetric_early_stop_150.pt")


# finetune
records = dcec.fit(cell_matrix,
                    num_labels,
                    lr=0.0001,
                    batch_size=4,
                    num_epochs=30,
                    update_interval=1,
                    tol=-1,
                    kmeans_init=True)

with open(save_file, 'w') as f:
    f.write(str(records))
