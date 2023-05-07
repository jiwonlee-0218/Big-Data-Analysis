import torch
# from torchmetrics.functional import pairwise_euclidean_distance


def compute_CE(x):
    """
    x shape : (n , n_hidden)
    return : output : (n , 1)
    """
    return torch.sqrt(torch.sum(torch.square(x[:, 1:] - x[:, :-1]), dim=1))




def compute_similarity(z, centroids, similarity="EUC"):   # z : (284, 10)
    """
    Function that compute distance between a latent vector z and the clusters centroids.
    similarity : can be in [CID,EUC,COR] :  euc for euclidean,  cor for correlation and CID
                 for Complexity Invariant Similarity.
    z shape : (batch_size, n_hidden)
    centroids shape : (n_clusters, n_hidden)
    output : (batch_size , n_clusters)
    """
    batch_size = z.shape[0]
    seq_length = z.shape[1]
    feature_size = z.shape[2]

    distance = torch.sqrt(torch.sum((z.unsqueeze(1) - centroids.unsqueeze(0)) ** 2, dim =-1))
    similarity = 1 - distance.mean(dim=-1)
    similarity = similarity.cpu().detach()
    return similarity

