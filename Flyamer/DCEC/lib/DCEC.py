import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from lib.utils import Dataset
from lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from lib.ops import MSELoss, BCELoss
import copy
import numpy as np
import math

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.decomposition import PCA


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1**(epoch // 300))
    toprint = True
    for param_group in optimizer.param_groups:
        if param_group["lr"] != lr:
            param_group["lr"] = lr
            if toprint:
                print("Switching to learning rate %f" % lr)
                toprint = False

class DCEC_BN_5(nn.Module):
    def __init__(self,
                input_shape=[250, 250, 1],
                embedding_dim=30,
                num_clusters=4,
                filters=[32, 64, 128, 64, 32],
                leaky=True,
                neg_slope=0.01,
                activations=False,
                bias=True):
        super(DCEC_BN_5, self).__init__()
        self.mu = nn.Parameter(torch.Tensor(num_clusters, 20 * embedding_dim))
        self.alpha = 1.0
        self.criterion = MSELoss()

        self.activations = activations
        # bias = True
        self.pretrained = False
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(input_shape[2],
                                filters[0],
                                5,
                                stride=2,
                                padding=2,
                                bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(filters[0],
                                filters[1],
                                5,
                                stride=2,
                                padding=2,
                                bias=bias)
        self.conv3 = nn.Conv2d(filters[1],
                                filters[2],
                                5,
                                stride=2,
                                padding=2,
                                bias=bias)
        self.conv4 = nn.Conv2d(filters[2],
                                filters[3],
                                5,
                                stride=2,
                                padding=2,
                                bias=bias)
        self.conv5 = nn.Conv2d(filters[3],
                                filters[4],
                                3,
                                stride=2,
                                padding=0,
                                bias=bias)

        # 卷积尺寸计算
        self.conv_sizes = []
        conv_size1 = (input_shape[0] + 1) // 2
        conv_size2 = (conv_size1 + 1) // 2
        conv_size3 = (conv_size2 + 1) // 2
        conv_size4 = (conv_size3 + 1) // 2
        conv_size5 = (conv_size4 - 1) // 2
        self.conv_sizes = [
            conv_size1, conv_size2, conv_size3, conv_size4, conv_size5
        ]
        self.lin_features_len = self.conv_sizes[-1] * \
            self.conv_sizes[-1] * filters[-1]
        self.embedding = nn.Linear(self.lin_features_len,
                                    embedding_dim,
                                    bias=bias)
        self.deembedding = nn.Linear(embedding_dim,
                                    self.lin_features_len,
                                    bias=bias)

        # 反卷积尺寸计算
        # H_out = (H_in - 1)*stride - 2*padding + kernel_size + output_padding
        # 所以
        # output_padding = H_out - (H_in - 1)*stride + 2*padding - kernel_size
        out_pad = self.conv_sizes[3] - (self.conv_sizes[4] - 1) * 2 + 2 * 0 - 3
        self.deconv5 = nn.ConvTranspose2d(filters[4],
                                            filters[3],
                                            3,
                                            stride=2,
                                            padding=0,
                                            output_padding=out_pad,
                                            bias=bias)
        out_pad = self.conv_sizes[2] - (self.conv_sizes[3] - 1) * 2 + 2 * 2 - 5
        self.deconv4 = nn.ConvTranspose2d(filters[3],
                                            filters[2],
                                            5,
                                            stride=2,
                                            padding=2,
                                            output_padding=out_pad,
                                            bias=bias)
        out_pad = self.conv_sizes[1] - (self.conv_sizes[2] - 1) * 2 + 2 * 2 - 5
        self.deconv3 = nn.ConvTranspose2d(filters[2],
                                            filters[1],
                                            5,
                                            stride=2,
                                            padding=2,
                                            output_padding=out_pad,
                                            bias=bias)
        out_pad = self.conv_sizes[0] - (self.conv_sizes[1] - 1) * 2 + 2 * 2 - 5
        self.deconv2 = nn.ConvTranspose2d(filters[1],
                                            filters[0],
                                            5,
                                            stride=2,
                                            padding=2,
                                            output_padding=out_pad,
                                            bias=bias)
        out_pad = self.input_shape[0] - (self.conv_sizes[0] -
                                         1) * 2 + 2 * 2 - 5
        self.deconv1 = nn.ConvTranspose2d(filters[0],
                                            input_shape[2],
                                            5,
                                            stride=2,
                                            padding=2,
                                            output_padding=out_pad,
                                            bias=bias)

        # BatchNorm2d
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.bn3_1 = nn.BatchNorm2d(filters[2])
        self.bn4_1 = nn.BatchNorm2d(filters[3])
        self.bn4_2 = nn.BatchNorm2d(filters[3])
        self.bn3_2 = nn.BatchNorm2d(filters[2])
        self.bn2_2 = nn.BatchNorm2d(filters[1])
        self.bn1_2 = nn.BatchNorm2d(filters[0])

        # ReLU copies for graph representation in tensorboard
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.check_code = True

    def encode(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        if self.check_code:
            print('conv1:', x.shape)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        if self.check_code:
            print('conv2:', x.shape)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        if self.check_code:
            print('conv3:', x.shape)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        if self.check_code:
            print('conv4:', x.shape)
        x = self.conv5(x)
        if self.activations:
            x = self.sig(x)
        else:
            x = self.relu5_1(x)
        if self.check_code:
            print('conv5:', x.shape)
        x = x.view(x.size(0), -1)
        if self.check_code:
            print('resize:', x.shape)
        x = self.embedding(x)
        if self.check_code:
            print('embed:', x.shape)
        z = x
        x = self.deembedding(x)
        x = self.relu5_2(x)
        if self.check_code:
            print('deembed:', x.shape)
        x = x.view(x.size(0), self.filters[-1], self.conv_sizes[-1], self.conv_sizes[-1])
        if self.check_code:
            print('resize:', x.shape)
        x = self.deconv5(x)
        x = self.relu4_2(x)
        x = self.bn4_2(x)
        if self.check_code:
            print('deconv5:', x.shape)
        x = self.deconv4(x)
        x = self.relu3_2(x)
        x = self.bn3_2(x)
        if self.check_code:
            print('deconv4:', x.shape)
        x = self.deconv3(x)
        x = self.relu2_2(x)
        x = self.bn2_2(x)
        if self.check_code:
            print('deconv3:', x.shape)
        x = self.deconv2(x)
        x = self.relu1_2(x)
        x = self.bn1_2(x)
        if self.check_code:
            print('deconv2:', x.shape)
        x = self.deconv1(x)
        if self.activations:
            x = self.tanh(x)
        if self.check_code:
            print('deconv1:', x.shape)
        self.check_code = False
        return z, x

    def forward(self, x):
        z = []
        x_recon = []
        for i in range(20):
            chrom_batch = x[:, i].unsqueeze(1) 
            z_i, x_i = self.encode(chrom_batch)
            z.append(z_i)
            x_recon.append(x_i)
        z = torch.cat(z, 1)
        x_recon = torch.cat(x_recon, 1)

        q = 1.0 / (1.0 + torch.sum(
            (z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q, x_recon

    def forwardBatch(self, x):
        z = []
        q = []
        x_recon = []
        for batch_idx, inputs in enumerate(x):
            # inputs = Variable(inputs)
            z_i, q_i, x_recon_i = self.forward(inputs.unsqueeze(0))
            z.append(z_i.data)
            q.append(q_i.data)
            x_recon.append(x_recon_i.data)
        z = torch.cat(z, dim=0)
        q = torch.cat(q, dim=0)
        x_recon = torch.cat(x_recon, dim=0)
        return z, q, x_recon

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.check_code = False
        pretrained_dict = torch.load(path,
                                    map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def KL_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(
                torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

# 三个画图的函数，服务器直接python .py跑的时候fit里相关代码注释掉。
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

    def plot_pca(self, epoch, z, labels):
        pca = PCA(n_components = 2)
        z_embedded = pca.fit_transform(z)

        if len(str(labels[0])) < 5:
            labels = np.load('path/to/cell_label_3.npy')

        cell_type = {0: 'oocyte',
                    1: 'pronucleus_male',
                    2: 'pronucleus_female'}


        cell_x = {}
        cell_y = {}
        for key in cell_type.values():
            cell_x[key] = []
            cell_y[key] = []
        for i in range(len(z_embedded)):
            cell_x[labels[i]].append(z_embedded[i][0])
            cell_y[labels[i]].append(z_embedded[i][1])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        cell_ax = []
        label_ax = []
        for key in cell_type.values():
            cell_ax.append(ax.scatter(cell_x[key], cell_y[key]))
            label_ax.append(key)

        ax.legend(cell_ax, label_ax, loc = 0)
        # ax.set_title('%d epoch: PCA' % epoch)
        ax.set_title('PCA')
        plt.show()

    def plot_tsne(self, epoch, z, labels):
        tsne = manifold.TSNE(n_components=2, init='pca')
        if z.shape[1] > 100:
            pca = PCA(n_components = 100)
            z= pca.fit_transform(z)
        z_embedded = tsne.fit_transform(z)

        if len(str(labels[0])) < 5:
            labels = np.load('path/to/cell_label_3.npy')


        cell_type = {0: 'oocyte',
                    1: 'pronucleus_male',
                    2: 'pronucleus_female'}

        cell_x = {}
        cell_y = {}
        for key in cell_type.values():
            cell_x[key] = []
            cell_y[key] = []
        for i in range(len(z_embedded)):
            cell_x[labels[i]].append(z_embedded[i][0])
            cell_y[labels[i]].append(z_embedded[i][1])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,7))
        cell_ax = []
        label_ax = []
        for key in cell_type.values():
            cell_ax.append(ax.scatter(cell_x[key], cell_y[key]))
            label_ax.append(key)

        ax.legend(cell_ax, label_ax, loc = 0)
        # ax.set_title('%d epoch: t-SNE' % epoch)
        ax.set_title('t-SNE')
        plt.show()

    def fit(self,
            X,
            y=None,
            lr=0.001,
            batch_size=32,
            num_epochs=10,
            update_interval=1,
            tol=1e-3,
            kmeans_init=True):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
            X = X.cuda()
            y = y.cuda()
        print("=====Training DCEC=======")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr)

        data = self.forwardBatch(X)[0].data.cpu().numpy()
        print('data', data.shape)
        y = y.cpu().numpy()

        # 初始化
        if kmeans_init == True:
            print("Initializing cluster centers with kmeans++")
            kmeans = KMeans(self.num_clusters, n_init=1000)
            y_pred = kmeans.fit_predict(data)
            self.plot_confusion_matrix(0, y_pred, y)
            self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
            print("Kmeans ACC: %.5f, NMI: %.5f, ARI: %.5f" %
                    (acc(y, y_pred), normalized_mutual_info_score(y, y_pred), 
                    adjusted_rand_score(y, y_pred)))

            # self.plot_pca(0, data, y)
            # self.plot_tsne(0, data, y)
        else:
            # random init experiment
            print("Randomly choose the centers")

        y_pred_last = y_pred

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        losses = []
        kl_losses = []
        recon_losses = []
        acces = []
        nmies = []
        aries = []
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                z, q, _ = self.forwardBatch(X)
                p = self.target_distribution(q).data

                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if y is not None:
                    acc_now = acc(y, y_pred)
                    nmi_now = normalized_mutual_info_score(y, y_pred)
                    ari_now = adjusted_rand_score(y, y_pred)
                    # print('z:')
                    # self.plot_pca(epoch, z.detach().cpu().numpy(), y)
                    # self.plot_tsne(epoch, z.detach().cpu().numpy(), y)
                    # print('q')
                    # self.plot_pca(epoch, q.detach().cpu().numpy(), y)
                    # self.plot_tsne(epoch, q.detach().cpu().numpy(), y)
                    # print('p:')
                    # self.plot_pca(epoch, p.detach().cpu().numpy(), y)
                    # self.plot_tsne(epoch, p.detach().cpu().numpy(), y)
                    
                    # self.plot_confusion_matrix(epoch, y_pred, y)

                    # print('z:')
                    # self.plot_pca(epoch, z.cpu().numpy(), y)
                    # self.plot_tsne(epoch, z.cpu().numpy(), y)
                    # print('q:')
                    # self.plot_pca(epoch, q.cpu().numpy(), y)
                    # self.plot_tsne(epoch, q.cpu().numpy(), y)
                    # print('p:')
                    # self.plot_pca(epoch, p.cpu().numpy(), y)
                    # self.plot_tsne(epoch, p.cpu().numpy(), y)

                    acces.append(acc_now)
                    nmies.append(nmi_now)
                    aries.append(ari_now)
                    print("epoch=%d: ACC: %.5f, NMI: %.5f, ARI: %.5f" %
                            (epoch, acc_now, nmi_now, ari_now))

                if tol > 0:
                    delta_label = np.sum(y_pred != y_pred_last).astype(
                        np.float32) / num
                    y_pred_last = y_pred
                    if epoch > 0 and delta_label < tol:
                        print("Reach tolerance threshold. Stop training")
                        break

            # train
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0

            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size:min((batch_idx + 1) *
                                                        batch_size, num)]
                pbatch = p[batch_idx * batch_size:min((batch_idx + 1) *
                                                        batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch, x_recon = self.forward(inputs)
                kl_loss = self.KL_loss(target, qbatch)
                recon_loss = self.criterion(inputs, x_recon)

                # loss选择，纯kl loss还是DCEC loss
                loss = 0.1 * kl_loss + recon_loss
                # loss = kl_loss

                train_loss += loss.data * len(inputs)
                train_recon_loss += recon_loss.data * len(inputs)
                train_kl_loss += kl_loss.data * len(inputs)

                loss.backward()
                optimizer.step()

            print("Loss: %.5f, Recon_Loss: %.5f, KL_Loss: %.5f" %
                    (train_loss / num, train_recon_loss / num,
                    train_kl_loss / num))
            losses.append(float((train_loss.detach().cpu() / num)))
            recon_losses.append(float((train_recon_loss.detach().cpu() / num)))
            kl_losses.append(float((train_kl_loss.detach().cpu() / num)))

        return losses, recon_losses, kl_losses, acces, nmies, aries
