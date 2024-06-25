import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
from kneed import KneeLocator
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans

def detect_reverb(burst_peaks, network_sdf, fs=12500):
    num_of_bins = int(len(burst_peaks) / 2.5)
    ibi = np.diff(burst_peaks)
    log_ibi = np.log(ibi)
    fr_at_peak = [network_sdf[(peak * fs).astype(int)] for peak in burst_peaks]
    placeholder = np.ones(len(log_ibi))
    #features = np.transpose([log_ibi, fr_at_peak[1:]])
    features = np.transpose([log_ibi, placeholder])

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 250,
        "random_state": 0
    }

    sse = []
    clusters = np.arange(1, 10, 1)
    for c in clusters:
        kmeans = KMeans(n_clusters=c, **kmeans_kwargs)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(clusters, sse, curve = "convex", direction = "decreasing")

    model = GaussianMixture(n_components=kl.elbow)
    #model = MiniBatchKMeans(n_clusters=2)

    yhat = model.fit_predict(features)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    '''# create scatter plot for samples from each cluster
    for cluster in clusters:
        # get row indexes for samples with this cluster
        row_ix = np.where(yhat == cluster)
        # create scatter of these samples
        plt.scatter(features[row_ix, 0], features[row_ix, 1])
    # show the plot
    plt.xscale('log')
    plt.show()'''

    cluster_0 = np.mean(features[np.where(yhat==0)], axis=0)
    cluster_1 = np.mean(features[np.where(yhat==1)], axis=0)

    dst = distance.euclidean(cluster_0, cluster_1)

    prime_idx = 1
    nonprime_idx = 0
    if np.mean(features[np.where(yhat==0), 0][0]) > np.mean(features[np.where(yhat==1), 0][0]):
        prime_idx = 0
        nonprime_idx = 1
    else:
        prime_idx = 1
        nonprime_idx = 0

    # Calculate IBPI overlap
    ibpi_0 = features[np.where(yhat==nonprime_idx), 0][0]
    ibpi_1 = features[np.where(yhat==prime_idx), 0][0]


    min_0 = min(ibpi_0)
    max_0 = max(ibpi_0)
    min_1 = min(ibpi_1)
    max_1 = max(ibpi_1)

    if (sum(ibpi_0 > min_1) + sum(ibpi_1 < max_0)) < (len(ibpi_0)+len(ibpi_1))*0.8:
        print(f"Network has clear different modes of bursting.")
        isReverb=True
    else:
        print(f"Network has too much overlap between burst frequencies.")
        isReverb=False

    return isReverb, features[:,0], features[:,1], yhat, prime_idx, nonprime_idx, dst

