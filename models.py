import torch.nn as nn
import torch
from utils import compute_similarity
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import gc
import glob
import os


class TAE_encoder(nn.Module):
    """
    Class for temporal autoencoder encoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    pooling : pooling number for maxpooling.
    """

    def __init__(self, filter_1, filter_lstm):
        super(TAE_encoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]


        ## CNN PART
        ### output shape (batch_size, 7 , n_hidden = 284)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=28,
                out_channels=filter_1,
                kernel_size=1,
                stride=1
            ),
            nn.LeakyReLU(),
        )


        ## LSTM PART
        ### output shape (batch_size , n_hidden = 284 , 50)
        self.lstm_1 = nn.LSTM(
            input_size= filter_1,
            hidden_size=self.hidden_lstm_1,
            batch_first=True

        )
        self.act_lstm_1 = nn.Tanh()


        ### output shape (batch_size , n_hidden = 284 , 10)
        self.lstm_2 = nn.LSTM(
            input_size=self.hidden_lstm_1,
            hidden_size=self.hidden_lstm_2,
            batch_first=True

        )
        self.act_lstm_2 = nn.Tanh()


    def forward(self, x):  # x : (1, 284, 116)

        ## encoder
        x = x.transpose(1,2)
        x = self.conv_layer_1(x)
        out_cnn = x.permute(0, 2, 1)

        out_lstm1, _ = self.lstm_1(out_cnn)
        out_lstm1_act = self.act_lstm_1(out_lstm1)


        features, _ = self.lstm_2(out_lstm1_act) # (1, 6, 64)
        out_lstm2_act = self.act_lstm_2(features)



        return out_lstm2_act


class TAE_decoder(nn.Module):
    """
    Class for temporal autoencoder decoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, filter_lstm):
        super(TAE_decoder, self).__init__()

        self.hidden_lstm_1 = filter_lstm[0]
        self.hidden_lstm_2 = filter_lstm[1]



        # upsample
        self.deconv_layer = nn.ConvTranspose1d(
            in_channels=self.hidden_lstm_2 ,
            out_channels=28,
            kernel_size=1,
            stride=1,
        )

    def forward(self, features):

        ## decoder
        features = features.transpose(1, 2)  ##(batch_size  , n_hidden , pooling)
        out_deconv = self.deconv_layer(features)
        out_deconv = out_deconv.permute(0,2,1)
        return out_deconv




class TAE(nn.Module):
    """
    Class for temporal autoencoder.
    filter_1 : filter size of the first convolution layer
    filter_lstm : hidden size of the lstm.
    """

    def __init__(self, args, filter_1=64,  filter_lstm=[50, 32]):  # 내 모델에 사용될 구성품을 정의 및 초기화하는 메서드
        super(TAE, self).__init__()


        self.filter_1 = filter_1
        self.filter_lstm = filter_lstm

        self.tae_encoder = TAE_encoder(
            filter_1=self.filter_1,
            filter_lstm=self.filter_lstm,
        )

        self.tae_decoder = TAE_decoder(
            filter_lstm = self.filter_lstm
        )



    def forward(self, x):        #  init에서 정의된 구성품들을 연결하는 메서드

        features = self.tae_encoder(x)
        out_deconv = self.tae_decoder(features)
        return features, out_deconv   # features는 clustering을 위해 encoder의 output을 사용














class ClusterNet(nn.Module):
    """
    class for the defintion of the DTC model
    path_ae : path to load autoencoder
    centr_size : size of the centroids = size of the hidden features
    alpha : parameter alpha for the t-student distribution.
    """

    def __init__(self, args, TAE: torch.nn.Module, cluster_centers = None, filter_lstm=[32, 8]):
        super(ClusterNet, self).__init__()



        ######### init with the pretrained autoencoder model
        # self.tae = TAE(args)


        self.tae = TAE ################ modified



        ## clustering model
        self.alpha_ = 1.0
        self.centr_size = filter_lstm[1]  # centroids_의 dimension
        self.n_clusters = args.n_clusters
        self.timeseries = args.serie_size
        self.device = args.device
        self.similarity = args.similarity

        ## centroid
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_clusters, self.timeseries, self.centr_size, dtype=torch.float            ## (8, 284, 116)
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers


        self.cluster_centers = nn.Parameter(initial_cluster_centers, requires_grad=True)


    def init_centroids(self, x):
        """
        This function initializes centroids with agglomerative clustering
        + complete linkage.
        """
        z, _ = self.tae(x)
        z_np = z.detach().cpu()

        km = KMeans(n_clusters=self.n_clusters, n_init=10).fit(z_np.reshape(z_np.shape[0], -1))
        cluster_centers = km.cluster_centers_.reshape(self.n_clusters, z_np.shape[1], z_np.shape[2])

        cluster_centers = torch.Tensor(cluster_centers).to(self.device)

        return cluster_centers



    def forward(self, x):

        z, x_reconstr = self.tae(x)  # z: (16, 284, 32), x_reconstr: (16, 284, 116)
        z_np = z.detach().cpu()

        similarity = compute_similarity(
            z, self.cluster_centers, similarity=self.similarity  # centroid와 representation간의 거리 계산
        )

        ## Q (batch_size , n_clusters)
        Q = torch.pow((1 + (similarity / self.alpha_)), -(self.alpha_ + 1) / 2)
        sum_columns_Q = torch.sum(Q, dim=1).view(-1, 1)
        Q = Q / sum_columns_Q

        ## P : ground truth distribution
        P = torch.pow(Q, 2) / torch.sum(Q, dim=0).view(1, -1)
        sum_columns_P = torch.sum(P, dim=1).view(-1, 1)
        P = P / sum_columns_P
        return z, x_reconstr, Q, P































