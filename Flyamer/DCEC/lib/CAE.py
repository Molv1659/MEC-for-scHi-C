import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from lib.utils import Dataset
from lib.ops import MSELoss, BCELoss
import copy

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch//300))
    toprint = True
    for param_group in optimizer.param_groups:
        if param_group["lr"] != lr:
            param_group["lr"] = lr
            if toprint:
                print("Switching to learning rate %f" % lr)
                toprint = False

class CAE_BN_5(nn.Module):
    def __init__(self, input_shape=[150, 150, 1], embedding_dim=30, num_clusters=4, filters=[32, 32, 64, 64, 128], leaky=True, neg_slope=0.01, activations=False, bias=True):
        super(CAE_BN_5, self).__init__()
        self.activations = activations
        # bias = True
        self.pretrained = False
        self.embedding_dim = embedding_dim
        self.num_clusters = num_clusters
        self.input_shape = input_shape
        self.filters = filters
        self.conv1 = nn.Conv2d(
            input_shape[2], filters[0], 5, stride=2, padding=2, bias=bias)
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope=neg_slope)
        else:
            self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            filters[0], filters[1], 5, stride=2, padding=2, bias=bias)
        self.conv3 = nn.Conv2d(
            filters[1], filters[2], 5, stride=2, padding=2, bias=bias)
        self.conv4 = nn.Conv2d(
            filters[2], filters[3], 5, stride=2, padding=2, bias=bias)
        self.conv5 = nn.Conv2d(
            filters[3], filters[4], 3, stride=2, padding=0, bias=bias)

        # 卷积尺寸计算
        self.conv_sizes = []
        conv_size1 = (input_shape[0] + 1) // 2
        conv_size2 = (conv_size1 + 1) // 2
        conv_size3 = (conv_size2 + 1) // 2
        conv_size4 = (conv_size3 + 1) // 2
        conv_size5 = (conv_size4 - 1) // 2
        self.conv_sizes = [conv_size1, conv_size2,
                            conv_size3, conv_size4, conv_size5]
        self.lin_features_len = self.conv_sizes[-1] * \
            self.conv_sizes[-1] * filters[-1]
        self.embedding = nn.Linear(
            self.lin_features_len, embedding_dim, bias=bias)
        self.deembedding = nn.Linear(
            embedding_dim, self.lin_features_len, bias=bias)

        # 反卷积尺寸计算
        # H_out = (H_in - 1)*stride - 2*padding + kernel_size + output_padding
        # 所以
        # output_padding = H_out - (H_in - 1)*stride + 2*padding - kernel_size
        out_pad = self.conv_sizes[3] - (self.conv_sizes[4]-1)*2 + 2*0 - 3
        self.deconv5 = nn.ConvTranspose2d(
            filters[4], filters[3], 3, stride=2, padding=0, output_padding=out_pad, bias=bias)
        out_pad = self.conv_sizes[2] - (self.conv_sizes[3]-1)*2 + 2*2 - 5
        self.deconv4 = nn.ConvTranspose2d(
            filters[3], filters[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = self.conv_sizes[1] - (self.conv_sizes[2]-1)*2 + 2*2 - 5
        self.deconv3 = nn.ConvTranspose2d(
            filters[2], filters[1], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = self.conv_sizes[0] - (self.conv_sizes[1]-1)*2 + 2*2 - 5
        self.deconv2 = nn.ConvTranspose2d(
            filters[1], filters[0], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)
        out_pad = self.input_shape[0] - (self.conv_sizes[0]-1)*2 + 2*2 - 5
        self.deconv1 = nn.ConvTranspose2d(
            filters[0], input_shape[2], 5, stride=2, padding=2, output_padding=out_pad, bias=bias)

        # BatchNorm2d 
        # 默认momentum=0.1
        # 改为0.001效果更好
        # self.bn1_1 = nn.BatchNorm2d(filters[0])
        # self.bn2_1 = nn.BatchNorm2d(filters[1])
        # self.bn3_1 = nn.BatchNorm2d(filters[2])
        # self.bn4_1 = nn.BatchNorm2d(filters[3])
        # self.bn4_2 = nn.BatchNorm2d(filters[3])
        # self.bn3_2 = nn.BatchNorm2d(filters[2])
        # self.bn2_2 = nn.BatchNorm2d(filters[1])
        # self.bn1_2 = nn.BatchNorm2d(filters[0])
        self.bn1_1 = nn.BatchNorm2d(filters[0], momentum = 0.001)
        self.bn2_1 = nn.BatchNorm2d(filters[1], momentum = 0.001)
        self.bn3_1 = nn.BatchNorm2d(filters[2], momentum = 0.001)
        self.bn4_1 = nn.BatchNorm2d(filters[3], momentum = 0.001)
        self.bn4_2 = nn.BatchNorm2d(filters[3], momentum = 0.001)
        self.bn3_2 = nn.BatchNorm2d(filters[2], momentum = 0.001)
        self.bn2_2 = nn.BatchNorm2d(filters[1], momentum = 0.001)
        self.bn1_2 = nn.BatchNorm2d(filters[0], momentum = 0.001)

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

        self.check_code = True # 检查代码用，会在第一次forward时输出各层后数据尺寸

    def forward(self, x):
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
        x = x.view(x.size(0), self.filters[-1],
                    self.conv_sizes[-1], self.conv_sizes[-1])
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

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.check_code = False
        pretrained_dict = torch.load(
            path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k,
                            v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def fit(self, chrom_matrix, labels, trainloader, validloader, lr=0.001, num_epochs=10, loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        # 存模型名称，一个存预训练指标最优，一个存early stop的，用后者进行后续细胞层级的finetune
        data_name = 'scHiCluster_data'
        save_cae_best_path = "model/" + data_name + "_cae_bn_5_lr_"+str(lr)[2:]+"_epoch_"+str(num_epochs)+'_'+loss_type+"_symmetric_"+str(self.input_shape[0])+"_best.pt"
        save_cae_path = "model/" + data_name + "_cae_bn_5_lr_"+str(lr)[2:]+"_epoch_"+str(num_epochs)+'_'+loss_type+"_symmetric_early_stop_"+str(self.input_shape[0])+".pt"
        print("save_cae_path: ", save_cae_path)

        # GPU检测
        use_cuda = torch.cuda.is_available()            
        if use_cuda:
            self.cuda()
            chrom_matrix_cuda = chrom_matrix.cuda()

        # 优化器，损失函数
        print("=====CAE Layer=======")
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        if loss_type == "mse":
            criterion = MSELoss()
        elif loss_type == "cross-entropy":
            criterion = BCELoss()

        # 初始化效果check
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs = self.forward(inputs)

            valid_recon_loss = criterion(outputs, inputs)
            total_loss += valid_recon_loss.data * len(inputs)
            total_num += inputs.size()[0]

        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))

        # start training
        best_ari = 0
        best_loss = 10000
        patience = 0
        for epoch in range(num_epochs):
            # train 1 epoch
            self.train()
            adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                # print("inputs.size:", inputs.size)
                # print("inputs.shape:", inputs.shape)
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)

                z, outputs = self.forward(inputs)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.data*len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate            
            # self.eval() 
            # 注释则预测时BN层动态更新，不注释则BN层预测时固定
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs = self.forward(inputs)

                valid_recon_loss = criterion(outputs, inputs)
                valid_loss += valid_recon_loss.data * len(inputs)

            cell_embedding = []
            for i in range(len(chrom_matrix)//20):
                z, x = self.forward(chrom_matrix_cuda[20*i:20*i+20])
                z = z.view(1, -1)
                cell_embedding.append(z.detach().cpu())
            cell_embedding = torch.cat(cell_embedding, 0)
            pca = PCA(n_components = 100)
            cell_embedding_reduce = pca.fit_transform(cell_embedding)
            y_pred = KMeans(n_clusters=self.num_clusters, n_init=2000).fit_predict(cell_embedding_reduce)
            score = adjusted_rand_score(labels, y_pred)

            print("#Epoch %3d: Reconstruct Loss: %.8f, Valid Reconstruct Loss: %.8f, ARI on all data: %.4f" % (
                epoch+1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset), score))

            # 存最优模型与判断early stop
            if score > best_ari:
                best_ari = score
                self.save_model(save_cae_best_path)
            if valid_loss / len(validloader.dataset) < best_loss:
                best_loss = valid_loss / len(validloader.dataset)
                patience = 0
            else:
                patience += 1
            if patience == 3:
                break
            
        
        print("best ari: %.3f" % best_ari)
        print("stop ari: %.3f" % score)
        self.save_model(save_cae_path)


            
            

            
