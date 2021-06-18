import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable


import numpy as np
import math
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.decomposition import PCA

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

class DEC(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
        encodeLayer=[400], activation="relu", dropout=0, alpha=1.0):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = Parameter(torch.Tensor(n_clusters, 23 * z_dim))


    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def forward(self, x, start=None, end=None):
        z = []
        for i in range(23):
            chrom_batch = x[:,i]
            h_i = self.encoder(chrom_batch)
            z_i = self._enc_mu(h_i)
            z.append(z_i)
        z = torch.cat(z, 1)
        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    def encodeBatch(self, dataloader, islabel=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        encoded = []
        ylabels = []
        self.eval()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z,_ = self.forward(inputs)
            encoded.append(z.data.cpu())
            ylabels.append(labels)

        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=1))
        
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

# 画图的函数，服务器直接python .py跑的时候fit里相关代码注释掉。
# 想看画图开个jupyter notebook跑，把fit里相关画图函数调用代码的注释去掉
    def plot_confusion_matrix(self, epoch, y_pred, y_true):
        # 自动对齐
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        from scipy.optimize import linear_sum_assignment as linear_assignment
        ind = linear_assignment(w.max() - w)
        ind = np.asarray(ind)
        ind = np.transpose(ind)
        trans_dict = {}
        for i,j in ind:
            trans_dict[i] = j
        y_pred_fix = [trans_dict[item] for item in y_pred]

        fig, ax = plt.subplots(figsize=(10,10))
        c2 = confusion_matrix(y_true, y_pred_fix)
        sns.heatmap(c2, annot=True, ax=ax)
        ax.set_title('%d epoch: confusion matrix' % epoch)
        ax.set_xlabel('predict') 
        ax.set_ylabel('true') 
        plt.show()

    def plot_q(self, epoch, q):
        sns.set()
        f, ax = plt.subplots()
        sns.heatmap(q, ax=ax)
        ax.set_title('%d epoch: soft assignment' % epoch)
        ax.set_xlabel('cell type') 
        ax.set_ylabel('cell_i') 

    def plot_pca(self, epoch, z, labels):
        pca = PCA(n_components = 2)
        z_embedded = pca.fit_transform(z)

        if len(str(labels[0])) < 5:
            trans = {0:'HeLa', 1:'HAP1', 2:'GM12878', 3:'K562'}
            labels = [trans[l] for l in labels]

        cell_type = {'HeLa': 0, 'HAP1': 1, 'GM12878': 2, 'K562': 3}

        cell_x = {}
        cell_y = {}
        for key in cell_type.keys():
            cell_x[key] = []
            cell_y[key] = []
        for i in range(len(z_embedded)):
            cell_x[labels[i]].append(z_embedded[i][0])
            cell_y[labels[i]].append(z_embedded[i][1])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        cell_ax = []
        label_ax = []
        for key in cell_type.keys():
            cell_ax.append(ax.scatter(cell_x[key], cell_y[key]))
            label_ax.append(key)

        ax.legend(cell_ax, label_ax, loc = 0)
        ax.set_title('PCA')
        plt.show()


    def plot_tsne(self, epoch, z, labels):
        tsne = manifold.TSNE(n_components=2, init='pca')
        if z.shape[1] > 100:
            pca = PCA(n_components = 100)
            z= pca.fit_transform(z)
        z_embedded = tsne.fit_transform(z)

        if len(str(labels[0])) < 5:
            trans = {0:'HeLa', 1:'HAP1', 2:'GM12878', 3:'K562'}
            labels = [trans[l] for l in labels]

        cell_type = {'HeLa': 0, 'HAP1': 1, 'GM12878': 2, 'K562': 3}

        cell_x = {}
        cell_y = {}
        for key in cell_type.keys():
            cell_x[key] = []
            cell_y[key] = []
        for i in range(len(z_embedded)):
            cell_x[labels[i]].append(z_embedded[i][0])
            cell_y[labels[i]].append(z_embedded[i][1])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        cell_ax = []
        label_ax = []
        for key in cell_type.keys():
            cell_ax.append(ax.scatter(cell_x[key], cell_y[key]))
            label_ax.append(key)

        ax.legend(cell_ax, label_ax, loc = 0)
        ax.set_title('t-SNE')
        plt.show()



    def fit(self, X, y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, kmeans_init=True):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            X = X.cuda()
            y = y.cuda()
        print("=====Training DEC=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        z, q = self.forward(X)
        print('z', z.shape)
        y = y.cpu().numpy()

        if kmeans_init == True:
            print("Initializing cluster centers with kmeans.") 
            kmeans = KMeans(self.n_clusters, n_init=2000)
            y_pred = kmeans.fit_predict(z.detach().cpu().numpy())
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
            print("Kmeans acc: %.5f, nmi: %.5f, ARI: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
            # self.plot_heatmap(0, y, y_pred)
            # self.plot_q(0, q.detach().cpu().numpy())
            # self.plot_pca(0, z.detach().cpu().numpy(), y)
            # self.plot_tsne(0, z.detach().cpu().numpy(), y)
        else:
            print("Randomly choose the centers")
            y_pred = np.random.randint(0, 4, y.shape)
            mu = []
            for i in range(4):
                index = np.argwhere(y_pred==i)
                data_i = data[index].squeeze()
                mu_i = torch.mean(data_i, 0, keepdim=True)
                mu.append(mu_i)
            mu = torch.cat(mu, axis=0)
            self.mu.data.copy_(mu)

        y_pred_last = y_pred
        
        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        kl_losses = []
        acces = []
        nmies = []
        aries = []
        for epoch in range(num_epochs):
            if epoch%update_interval == 0:
                # update the targe distribution p
                z, q = self.forward(X)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if y is not None:
                    print("acc: %.5f, nmi: %.5f, ARI: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), adjusted_rand_score(y, y_pred)))
                    # self.plot_heatmap(epoch, y, y_pred)
                    # self.plot_q(epoch,q.detach().cpu().numpy())
                    # print('z:')
                    # self.plot_pca(epoch, z.detach().cpu().numpy(), y)
                    # self.plot_tsne(epoch, z.detach().cpu().numpy(), y)
                    # print('q')
                    # self.plot_pca(epoch, q.detach().cpu().numpy(), y)
                    # self.plot_tsne(epoch, q.detach().cpu().numpy(), y)
                    acces.append(acc(y, y_pred))
                    nmies.append(normalized_mutual_info_score(y, y_pred))
                    aries.append(adjusted_rand_score(y, y_pred))

                # check stop criterion
                if tol > 0:
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                    y_pred_last = y_pred
                    if epoch>0 and delta_label < tol:
                        print('delta_label ', delta_label, '< tol ', tol)
                        print("Reach tolerance threshold. Stopping training.")
                        break
                

            # train 
            train_loss = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                pbatch = p[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
                
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch = self.forward(inputs, batch_idx*batch_size, min((batch_idx+1)*batch_size, num))
                loss = self.loss_function(target, qbatch)
                train_loss += loss.data*len(inputs)
                loss.backward()
                optimizer.step()

            print("#Epoch %3d: Loss: %.4f" % (
                epoch+1, train_loss / num))
            kl_losses.append(float(train_loss.detach().cpu() / num))
            
        return kl_losses, acces, nmies, aries




