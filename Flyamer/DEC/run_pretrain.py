import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch.nn.functional as nnf
from sklearn.model_selection import train_test_split

from lib.stackedDAE import StackedDAE
from lib.dec import DEC

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

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
chrom_matrix = torch.cat(chrom_matrix, 0).view(chrom_num, 150*150)

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

# pretrain stacked autoencoder
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sdae = StackedDAE(input_dim=150*150, z_dim=30, binary=False,
    encodeLayer=[300], decodeLayer=[300], activation="relu", 
    dropout=0)
print(sdae)

sdae.pretrain(train_loader, val_loader, lr=0.00002, batch_size=batch_size, 
    num_epochs=100, corrupt=0.2, loss_type="mse")
sdae.fit(train_loader, val_loader, lr=0.00002, num_epochs=200, corrupt=0.2, loss_type="mse")
sdae_savepath = ("model/sdae_300_30_lr_00002_epoch_100_Adam_symmetric_scHiCluster_data.pt")
sdae.save_model(sdae_savepath)



# 初始KMeans效果
cell_embedding = []
labels = np.load('path/to/cell_label_num.npy')

use_cuda = torch.cuda.is_available()            
if use_cuda:
    sdae.cuda()
    chrom_matrix_cuda = chrom_matrix.cuda()
    
for i in range(len(chrom_matrix)//20):
    z, x = sdae.forward(chrom_matrix_cuda[20*i:20*i+20])
    z = z.view(1, -1)
    cell_embedding.append(z.detach().cpu())
cell_embedding = torch.cat(cell_embedding, 0)
pca = PCA(n_components = 100)
cell_embedding_reduce = pca.fit_transform(cell_embedding)
y_pred = KMeans(n_clusters=3, n_init=2000).fit_predict(cell_embedding_reduce)

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

acc = acc(labels, y_pred)
nmi = normalized_mutual_info_score(labels, y_pred)
ari = adjusted_rand_score(labels, y_pred)
print("ACC: %.5f, NMI: %.5f, ARI: %.5f" %(acc, nmi, ari))

