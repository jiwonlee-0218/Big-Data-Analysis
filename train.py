import torch
import datetime
import torch.nn as nn
import os
import numpy as np
from main_config import get_arguments
from models import ClusterNet, TAE
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn import metrics


from tslearn.clustering import TimeSeriesKMeans


def writelog(file, line):
    file.write(line + '\n')
    print(line)



def pretrain_autoencoder(args, verbose=True, directory='.'):
    """
    function for the autoencoder pretraining
    """


    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.exists(os.path.join(directory, 'models_logs')):
        os.makedirs(os.path.join(directory, 'models_logs'))

    if not os.path.exists(os.path.join(directory, 'models_weights')):
        os.makedirs(os.path.join(directory, 'models_weights'))

    # Text Logging
    f = open(os.path.join(directory, 'setting.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'GPU ID: %s' % (args.gpu_id))
    writelog(f, 'Dataset: %s' % (args.dataset))
    writelog(f, '----------------------')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Max Patience: %d (10 percent of the epoch size)' % args.max_patience)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    f.close()



    print("Pretraining autoencoder... \n")
    writer = SummaryWriter(log_dir=os.path.join(directory, 'models_logs'))

    ## define TAE architecture
    tae = TAE(args, filter_1=56,  filter_lstm=[32, 8])
    tae = tae.to(args.device)
    print(tae)

    ## MSE loss
    loss_ae = nn.MSELoss()
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    for epoch in range(args.epochs_ae):

        # training
        tae.train()
        all_loss = 0


        for batch_idx, (inputs, _) in enumerate(train_dl):
            inputs = inputs.type(torch.FloatTensor).to(args.device)

            optimizer.zero_grad()  # 기울기에 대한 정보 초기화
            features, x_reconstr = tae(inputs)
            loss_mse = loss_ae(inputs, x_reconstr)  # x_reconstr(decoded) & 원본(input) 사이의 평균제곱오차
            loss_mse.backward()  # 기울기 구함


            optimizer.step()  # 최적화 진행

            all_loss += loss_mse.item()

        train_loss = all_loss / (batch_idx + 1)

        writer.add_scalar("training loss", train_loss, epoch+1)
        if verbose:
            print("Pretraining autoencoder loss for epoch {} is : {}".format(epoch + 1, train_loss))

        # validation
        tae.eval()
        with torch.no_grad():
            all_val_loss = 0
            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)
                v_features, val_reconstr = tae(val_x)
                val_loss = loss_ae(val_x, val_reconstr)

                all_val_loss += val_loss.item()

            validation_loss = all_val_loss / (j + 1)

            writer.add_scalar("validation loss", validation_loss, epoch+1)
            print("val_loss for epoch {} is : {}".format(epoch + 1, validation_loss))

        if epoch == 0:
            min_val_loss = validation_loss

        if validation_loss < min_val_loss:
            torch.save({
                'model_state_dict': tae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': validation_loss
            }, os.path.join(directory, 'models_weights') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch + 1,
                                                                                                        validation_loss))
            min_val_loss = validation_loss
            print("save weights !!")

    writer.close()
    print("Ending pretraining autoencoder. \n")







def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))



def train_ClusterNET(epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    train_loss = 0



    for batch_idx, (inputs, labels) in enumerate(train_dl):  # epoch 1에 모든 training_data batch만큼
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        optimizer.zero_grad()

        z, x_reconstr, Q, P = model(inputs)  # ClusterNet의 forward

        loss_mse = loss_ae(inputs, x_reconstr)
        loss_KL = kl_loss_function(P, Q)
        warnings.filterwarnings("ignore")


        total_loss = loss_mse + loss_KL
        total_loss.backward()
        optimizer.step()

        train_loss += total_loss.item()



    return  (train_loss / (batch_idx + 1))







def initalize_centroids(X):
    """
    Function for the initialization of centroids.
    """
    X_tensor = X.type(torch.FloatTensor).to(args.device)
    init_centroids = model.init_centroids(X_tensor)
    return init_centroids




def training_function(args, X_test, y_test, verbose=True):

    """
    function for training the DTC network.
    """


    ## train clustering model

    print("Training full model ...")
    if not os.path.exists(os.path.join(directory, 'full_models')):
        os.makedirs(os.path.join(directory, 'full_models'))  ### for model save directory
    if not os.path.exists(os.path.join(directory, 'full_models_logs')):
        os.makedirs(os.path.join(directory, 'full_models_logs'))
    writer = SummaryWriter(log_dir=os.path.join(directory, 'full_models_logs'))

    f = open(os.path.join(os.path.join(directory, 'full_models/'), 'DTC Model\'s ARI.log'), 'a')
    writelog(f, '======================')
    writelog(f, 'Model Name: %s' % args.model_name)
    writelog(f, '----------------------')
    writelog(f, 'Epoch: %d' % args.epochs_ae)
    writelog(f, 'Batch Size: %d' % args.batch_size)
    writelog(f, 'Learning Rate: %s' % str(args.lr_ae))
    writelog(f, 'Weight Decay: %s' % str(args.weight_decay))
    writelog(f, '======================')
    writelog(f, 'If the validation loss decreases compared to the previous epoch...')
    f.close()






    cluster_centers = initalize_centroids(X_train)  ##########################################################
    model.state_dict()['cluster_centers'].copy_(cluster_centers)  ## 모델 초기화할 때 initial centroid 전에 xavier한 것 -> 위에 정의한 cluster_centers로 바뀌었다.





    for epoch in tqdm(range(args.max_epochs)):
        train_loss = train_ClusterNET(epoch, args, verbose=verbose) # 1 epoch training
        print("For epoch : ", epoch, "Total Loss is : %.4f" % (train_loss))

        writer.add_scalar("training total loss", train_loss, (epoch))


        model.eval()
        with torch.no_grad():
            all_val_loss = 0


            for j, (val_x, val_y) in enumerate(valid_dl):
                val_x = val_x.type(torch.FloatTensor).to(args.device)



                z, x_reconstr, Q, P = model(val_x)

                V_loss_mse = loss_ae(val_x, x_reconstr)
                V_loss_KL = kl_loss_function(P, Q)
                V_total_loss = V_loss_mse + V_loss_KL


                all_val_loss += V_total_loss.item()





            all_val_loss = all_val_loss / (j+1)
            print("For epoch: ", epoch, "val_Total Loss is : %.4f" % (all_val_loss))
            writer.add_scalar("validation total loss", all_val_loss, (epoch))


        torch.save(
            model,
            os.path.join(directory, 'full_models') + '/checkpoint_epoch_{}_loss_{:.5f}.pt'.format(epoch,  all_val_loss)
        )



    writer.close()

    all_test_preds, all_test_gt = [], []
    X_test = X_test.type(torch.FloatTensor).to(args.device)
    z, x_reconstr, Q, P = model(X_test)
    all_test_gt.append(y_test.cpu().detach())


    preds = torch.max(Q, dim=1)[1]
    all_test_preds.append(preds.cpu().detach())


    all_test_gt = torch.cat(all_test_gt, dim=0).numpy()
    all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
    accuracy = metrics.accuracy_score(all_test_gt, all_test_preds)
    print("test accuracy is : %.4f" % (accuracy))
    print("Ending Training full model... \n")







ucr = UCR_UEA_datasets()
all_ucr_datasets = ucr.list_datasets()

def load_ucr(dataset='CBF'):
    X_train, y_train, X_test, y_test = ucr.load_dataset(dataset)
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))
    if dataset == 'HandMovementDirection':  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(y)  # sometimes labels are strings or start from 1
    assert(y.min() == 0)  # assert labels are integers and start from 0
    # preprocess data (standardization)
    X_scaled = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X_scaled, y


def load_data(dataset_name):
    if dataset_name in all_ucr_datasets:
        return load_ucr(dataset_name)
    else:
        print('Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.'.format(dataset_name))
        exit(0)


if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()

    # GPU Configuration
    gpu_id = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args.device)


    # data load
    (X_train, y_train) = load_data(args.dataset)



    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=42, test_size=0.3)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

    X_train, y_train = torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    X_val, y_val = torch.FloatTensor(X_val), torch.FloatTensor(y_val)
    X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size)  ###
    valid_ds = TensorDataset(X_val, y_val)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size)  ###




    directory = os.path.join(args.dir_root, args.model_name, args.dataset,
                             'Epochs' + str(args.epochs_ae) + '_BS_' + str(args.batch_size) + '_LR_' + str(
                                 args.lr_ae) + '_wdcay_' + str(args.weight_decay))

    # number of clusters
    args.n_clusters = len(np.unique(y_train))
    # timeseries
    args.serie_size = X_train.shape[1]








    if args.ae_weights is None and args.epochs_ae > 0:  ########### pretrain

        pretrain_autoencoder(args, directory=directory)





    if args.ae_weights is not None and args.ae_models is None:  #### weight ok, but model is none


        weight_path = os.path.join(directory, args.ae_weights)
        full_path = sorted(glob.glob(weight_path + '*'), key=os.path.getctime)
        full_path = full_path[-1]
        print(full_path)

        tae = TAE(args, filter_1=56,  filter_lstm=[32, 8])
        checkpoint = torch.load(full_path, map_location=args.device)
        tae.load_state_dict(checkpoint['model_state_dict'])

        tae = tae.to(args.device)
        print(tae)

        ## MSE loss
        loss_ae = nn.MSELoss()
        ## Optimizer
        optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        model = ClusterNet(args, tae)  ## 모델 초기화 with the pretrained autoencoder model
        model = model.to(args.device)

        training_function(args, X_test, y_test)






















''' multivariate timeseries kmeans clustering method'''
# def initalize_centroids(X):
#     """
#     Function for the initialization of centroids.
#     """
#
#     tae = model.tae
#     tae = tae.to(args.device)
#     X_tensor = X.type(torch.FloatTensor).to(args.device)
#
#     X_tensor =  X_tensor.detach()
#     z, x_reconstr = tae(X_tensor)
#     print('initialize centroid')
#
#     features = z.detach().cpu()  # z, features: (864, 284, 32)
#
#     km = TimeSeriesKMeans(n_clusters=args.n_clusters, verbose=False, random_state=42)
#     assignements = km.fit_predict(features)
#     # assignements = AgglomerativeClustering(n_clusters= args.n_clusters, linkage="complete", affinity="euclidean").fit(features)
#     # km.inertia_
#     # assignements (864,)
#     # km.cluster_centers_   (8, 284, 32)
#
#     centroids_ = torch.zeros(
#         (args.n_clusters, z.shape[1], z.shape[2]), device=args.device
#     )  # centroids_ : torch.Size([8, 284, 32])
#
#     for cluster_ in range(args.n_clusters):
#         centroids_[cluster_] = features[assignements == cluster_].mean(axis=0)
#     # centroids_ : torch.Size([8, 284, 32])
#
#     cluster_centers = centroids_
#
#     return cluster_centers