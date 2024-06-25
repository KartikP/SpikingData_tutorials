import numpy as np
import pandas as pd
import math
from burst_detection import maxInterval
from itertools import chain
import matplotlib.pyplot as plt
from itertools import cycle, chain
from scipy.stats import norm
from scipy.signal import convolve
from scipy.signal import find_peaks
import random
from sklearn.cluster import KMeans

def organize_axion_spikelist(filepath):
    # Create neuralMetric dataframe
    spikelist_file = pd.read_csv(filepath, sep='\r\n', header=None, engine="python")
    spikelist_file = spikelist_file[0].str.split(',', expand=True)
    spikelist_file.columns = spikelist_file.iloc[0]
    spikelist_file = spikelist_file.iloc[1:]

    spiketimes = spikelist_file[["Time (s)", "Electrode"]]
    spiketimes = spiketimes.iloc[:-8]
    spiketimes = spiketimes.dropna()
    spiketimes = spiketimes[spiketimes["Electrode"].str.contains("_")]
    spiketimes = spiketimes.astype({"Time (s)": float})
    spiketimes["Well"] = [electrode.split("_")[0] for electrode in list(spiketimes["Electrode"].values)]
    spiketimes["Channel"] = [electrode.split("_")[1] for electrode in list(spiketimes["Electrode"].values)]

    r,c = np.where(spikelist_file == "Well Information")
    metadata_slice = spikelist_file.iloc[r[0]:].transpose().reset_index()
    metadata_slice.columns = metadata_slice.iloc[0]
    metadata_slice = metadata_slice.iloc[1:]
    metadata = {}
    label = []
    well_details = {}
    for i, well in enumerate(metadata_slice["Well"]):
        if well != None:
            metadata[well] = metadata_slice.iloc[i]["Treatment"]
            label.append(str(well)+"-"+str(metadata_slice.iloc[i]["Treatment"]))
            detail_tmp = metadata_slice[metadata_slice["Well"]==well]
            well_details[well] = ([detail_tmp["Well"].values[0], detail_tmp["Treatment"].values[0], detail_tmp["Concentration"].values[0], detail_tmp["Additional Information"].values[0]])

    duration = int(math.ceil(max(spiketimes["Time (s)"]) / 100.0)) * 100

    return spiketimes, metadata, label, well_details, duration


def generate_gaussian_kernel(sigma=0.1, fs=12500):
    '''
    :param sigma: Width of kernel
    :param fs: Sampling frequency in Hz
    :return: Gaussian kernel
    '''
    edges = np.arange(-3 * sigma, 3 * sigma, 1 / fs)
    kernel = norm.pdf(edges, 0, sigma)
    kernel = kernel / max(kernel)
    return kernel


def generate_spikematrix(spiketrain, fs=12500, duration=300):
    '''
    :param spiketrain: Takes spikes times from a single channel
    :param fs: Sampling frequency in Hz
    :param duration: Duration of recording in seconds
    :return: spikematrix: Binary matrix containing spikes
    '''
    spiketimes = np.array(spiketrain)
    spiketimes = spiketimes[spiketimes <= duration]  # Ensure recording is desired length
    spikematrix = [0] * (duration * fs)  # Generate empty spike matrix with appropriate number of bins
    for spike in spiketimes:
        spikematrix[int(spike * fs)] = 1
    return spikematrix


def generate_sdf(spikematrix, gaussian_kernel):
    '''
    :param spikematrix: Binary matrix containing spikes
    :param gaussian_kernel: Gaussian kernel to be convolved with spikematrix
    :return: sdf: Continuous timeseries representing probability distribution of activity
    '''
    sdf_tmp = convolve(spikematrix, gaussian_kernel)
    sdf_tmp = sdf_tmp + min(sdf_tmp)
    sdf = sdf_tmp[int((len(sdf_tmp)-len(spikematrix))/2):int(len(sdf_tmp)-((len(sdf_tmp)-len(spikematrix))/2))]
    sdf = sdf/max(gaussian_kernel)
    return sdf

def detect_burst_peaks(sdf, height = 0.5, prom=0.5, fs=12500):
    burst_peaks, _ = find_peaks(sdf, height=height, prominence=prom, distance=5)
    return burst_peaks/fs

def generate_raster(spiketimes, well_id, duration, channel_ids):
  '''
  well_data: Data frame in the format seen above
  duration: Length of recording in seconds (in this case 300 seconds)
  channel_ids: Name of all channels in the format seen in the data frame "channel" column
  '''
  raster = []
  # Loop through each channel regardless of activity
  well_spiketimes = spiketimes[spiketimes["Well"] == well_id].reset_index(drop=True)
  for channel in channel_ids:
    # If the channel is found in my data (i.e., it had at least 1 spike)
    if str(channel) in well_spiketimes["Channel"].to_list():
      # Find spike times associated with that channel
      spiketrain = well_spiketimes.loc[well_spiketimes["Channel"]==str(channel), well_spiketimes.columns.str.contains("Time")].reset_index(drop=True)
      # Convert spike times that occur before the end of the recording (i.e., duration) to an array
      spiketrain = spiketrain.to_numpy().flatten()[spiketrain.to_numpy().flatten() <= duration]
      # Store array
      raster.append(spiketrain)
    # If the channel is not found in my data, I still want to store that it was empty
    else:
      raster.append([])
  return raster

def rasterList2Scatter(raster):
    tmp = []
    for i, well in enumerate(raster):
        channel_tmp = np.ones(len(raster[i]))*i
        if len(channel_tmp) > 0:
           tmp.extend(list(zip(channel_tmp, raster[i])))
    channel = [spike[0] for spike in tmp]
    spiketime = [spike[1] for spike in tmp]
    return channel, spiketime


'''
BURST DETECTION
'''
def detect_bursts(well, max_begin_ISI=0.17, max_end_ISI=0.2, min_IBI=0.2, min_burst_duration=0.01,
                  min_spikes_in_burst=3):
    all_burst_data = {}
    for i, ch in enumerate(well):
        channel_burst_data, tooShort = maxInterval(ch, max_begin_ISI, max_end_ISI, min_IBI, min_burst_duration,
                                                   min_spikes_in_burst)
        all_burst_data[i] = channel_burst_data
    return all_burst_data

def generate_burst_raster(all_burst_data):
    burst_raster = []
    for ch in range(len(all_burst_data)):
        channel_bursts = all_burst_data[ch]
        if len(channel_bursts) > 0:
            channel_bursting_spikes = list(chain.from_iterable(channel_bursts.values()))
            burst_raster.append(np.array(channel_bursting_spikes))
        else:
            burst_raster.append([])
    return burst_raster


'''
FEATURE CALCULATIONS
'''
'''
Spike train features
'''
def calculate_mFR(nSpikes, duration):
    return (np.sum(nSpikes) / duration)/ len(nSpikes)


def calculate_isi(well):
    isi = []
    for ch in well:
        isi = np.append(isi, np.diff(ch))
    return isi

def calculate_isiCV(well):
    isi_cv = []
    for ch in well:
        if len(ch) > 5:
            isi_cv = np.append(isi_cv, (np.nanstd(np.diff(ch)) / np.nanmean(np.diff(ch))))
        else:
            isi_cv = np.append(isi_cv, np.nan)
    return isi_cv, np.nanmean(isi_cv)


def calculate_nActiveElectrodes(nSpikes, criteria=(100, 2)):
    return len(nSpikes[nSpikes > criteria[0]])


def calculate_wmFR(nSpikes, duration, nActiveElectrodes):
    wmfr_tmp = (np.sum(nSpikes) / duration) / nActiveElectrodes
    if (wmfr_tmp == np.inf) | (wmfr_tmp == -np.inf):
        return np.nan
    else:
        return wmfr_tmp


def calculate_nBursts(all_burst_data):
    return (sum([len(value) for value in all_burst_data.values()]), ([len(value) for value in all_burst_data.values()]),
            np.nanmean([len(value) for value in all_burst_data.values()]))


def calculate_nBurstingElectrodes(all_burst_data, criteria=(250,2)):
    return sum([len(value) >= criteria[1] for value in all_burst_data.values()])

def adaptationIndex(spiketrain):
    '''
    :param spiketrain:
    :return: adaptation index:
    A will be between -1 and 1.
    Positive for decelerating spike trains.
    Negative for accelerating spike trains.
    '''
    isi = np.diff(spiketrain)
    A = 0
    for i in range(len(isi)-1):
        A += ((isi[i+1]-isi[i])/(isi[i+1]+isi[i]))
    return A/(len(isi)-1)

def calculate_adaptationIndex(all_burst_data):
    A = []
    for ch in all_burst_data.values():
        A = np.append(A, [adaptationIndex(value) for value in ch.values()])
    return A, np.nanmean(A), np.nanstd(A)


'''
Burst features
'''
def calculate_burstDuration(all_burst_data):
    burst_duration = []
    for ch in all_burst_data.values():
        burst_duration = np.append(burst_duration, [(value[1] - value[0]) for value in ch.values()])
    return burst_duration, np.nanmean(burst_duration), np.nanstd(burst_duration)


def calculate_nSpikesPerBurst(all_burst_data):
    nSpikesPerBurst = []
    for ch in all_burst_data.values():
        nSpikesPerBurst = np.append(nSpikesPerBurst, [len(value) for value in ch.values()])
    return nSpikesPerBurst, np.nanmean(nSpikesPerBurst), np.nanstd(nSpikesPerBurst)

def calculate_burstISI(all_burst_data):
    burstISI = []
    for ch in all_burst_data.values():
        burstISI = np.append(burstISI, [np.nanmean(np.diff(value)) for value in ch.values()])
    return burstISI, np.nanmean(burstISI), np.nanstd(burstISI)

def calculate_ISIWithinBurst(all_burst_data):
    ISIWithinBurst = []
    for ch in all_burst_data.values():
        channelISIWithinBurst = [np.diff(value) for value in ch.values()]
        ISIWithinBurst.append(channelISIWithinBurst)
    return ISIWithinBurst, list(chain.from_iterable(ISIWithinBurst))


def calculate_ISIWithinBurst_average(all_burst_data, type="mean"):
    ISIWithinBurst = []
    if type == "mean":
        for ch in all_burst_data.values():
            if len(ch) > 3:
                channelISIWithinBurst = [np.diff(value) for value in ch.values()]
                ISIWithinBurst = np.append(ISIWithinBurst, np.nanmean(
                    [ISIWithinBurst for burst in channelISIWithinBurst for ISIWithinBurst in burst]))
        return np.nanmean(ISIWithinBurst), np.nanstd(ISIWithinBurst)
    elif type == "median":
        for ch in all_burst_data.values():
            if len(ch) > 3:
                channelISIWithinBurst = [np.diff(value) for value in ch.values()]
                ISIWithinBurst = np.append(ISIWithinBurst, np.nanmedian(
                    [ISIWithinBurst for burst in channelISIWithinBurst for ISIWithinBurst in burst]))
        return np.nanmean(ISIWithinBurst), np.nanstd(ISIWithinBurst)


def calculate_IBI(all_burst_data, criteria=(250,2)):
    IBIPerChannel = []
    for ch in all_burst_data.values():
        if len(ch) > criteria[1]:
            IBIPerChannel = np.append(IBIPerChannel,
                                      np.nanmean([ch[value][0] - ch[value - 1][-1] for value in range(1, len(ch))]))
    return IBIPerChannel, np.nanmean(IBIPerChannel), np.nanstd(IBIPerChannel)

def calculate_IBICV(all_burst_data, criteria=(250,2)):
    IBIPerChannel_CV = []
    for ch in all_burst_data.values():
        if len(ch) > criteria[1]:
            IBIPerChannel_CV = np.append(IBIPerChannel_CV,
                                         np.nanstd([ch[value][0] - ch[value - 1][-1] for value in range(1, len(ch))]) /
                                         np.nanmean([ch[value][0] - ch[value - 1][-1] for value in range(1, len(ch))]))
    return np.nanmean(IBIPerChannel_CV), np.nanstd(IBIPerChannel_CV)

'''
Network features
'''
def reverbDetection(network_sdf, min_ibi=0.2):
    '''
    INCOMPLETE
    :param network_sdf:
    :param min_ibi:
    :return:
    '''
    min_ibi_ind = int(min_ibi * 12500)

    burst_peaks, _ = find_peaks(network_sdf, height=0.5, distance=min_ibi_ind, prominence=0.5)
    burst_heights = network_sdf[burst_peaks]
    ibi = np.diff(burst_peaks / 12500)
    log_ibi = np.log(ibi)

    features = np.transpose([log_ibi, burst_heights[1:]])
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 250,
        "random_state": 0
    }

    kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
    kmeans.fit(features)

    x = np.transpose(features)[0]
    y = np.transpose(features)[1]

    cluster_center_x = np.transpose(kmeans.cluster_centers_)[0, :]
    cluster_center_y = np.transpose(kmeans.cluster_centers_)[1, :]

    # Determine which cluster label belongs to prime peaks
    if (cluster_center_y[0] > cluster_center_y[1]) & (cluster_center_x[0] > cluster_center_y[1]):
        cluster_x_1 = x[kmeans.labels_ == 0]
        cluster_x_2 = x[kmeans.labels_ == 1]
        cluster_y_1 = y[kmeans.labels_ == 0]
        cluster_y_2 = y[kmeans.labels_ == 1]
        kmeans.labels_ = kmeans.labels_ + 1
    else:
        cluster_x_1 = x[kmeans.labels_ == 1]
        cluster_x_2 = x[kmeans.labels_ == 0]
        cluster_y_1 = y[kmeans.labels_ == 1]
        cluster_y_2 = y[kmeans.labels_ == 0]

    amax = min(cluster_y_1)
    rmax = min(cluster_x_1)

    return

def networkBurstDetection_hist(network_sdf, nRandomSamples=10000, rmax = 0.1, fs=12500):
    randomSample = random.sample(list(network_sdf), nRandomSamples)
    noise_threshold = np.median(randomSample)
    nb_threshold = noise_threshold + 2

    min_IBI = rmax
    if min_IBI <= 10:
        min_IBI = 10
    min_duration = 0.01

    above_threshold = network_sdf >= nb_threshold

    # Find the indices where there are changes in the above_threshold state
    change_indices = np.where(np.diff(above_threshold))[0] + 1

    # Separate start and end indices
    starts = change_indices[above_threshold[change_indices - 1] == False]  # transition from False to True
    ends = change_indices[above_threshold[change_indices - 1] == True]  # transition from True to False

    # Handle cases where the series starts or ends above the threshold
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(network_sdf))

    # Combine starts and ends into a list of tuples
    peak_intervals = list(zip(starts/fs, ends/fs))

    # Merge close intervals
    i = 0
    while i < len(peak_intervals) - 1:
        # Check if the gap between current end and next start is less than thresholdY
        if peak_intervals[i + 1][0] - peak_intervals[i][1] < min_IBI:
            # Merge intervals
            peak_intervals[i] = (peak_intervals[i][0], peak_intervals[i + 1][1])
            del peak_intervals[i + 1]
        else:
            i += 1

    # Filter out short peaks
    nb_boundaries = [interval for interval in peak_intervals if (interval[1] - interval[0]) >= min_duration]

    return nb_boundaries

def countChannelReverberations(network_burst_start, network_burst_end, all_burst_data):
    all_channel_reverbs = {}
    mean_channel_reverbs = []
    median_channel_reverbs = []
    for c in range(len(all_burst_data)):
        channel_data = all_burst_data[c]
        middle_of_burst = []
        for b in channel_data:
            burst = channel_data[b]
            middle_of_burst.append(burst[int(len(burst)/2)])
        channel_reverbs = []
        for n in range(len(network_burst_start)):
            start = network_burst_start[n]
            end = network_burst_end[n]
            channel_reverbs.append(sum((middle_of_burst >= start) & (middle_of_burst <= end)) - 1)
        all_channel_reverbs[c] = channel_reverbs
        mean_channel_reverbs.append(np.nanmean([[r for r in channel_reverbs if r > 0]]))
        median_channel_reverbs.append(np.nanmedian([[r for r in channel_reverbs if r > 0]]))

    return all_channel_reverbs, mean_channel_reverbs, median_channel_reverbs


def nRSBs(all_channel_reverbs, threshold=0.3):
    channel_reverbs = [list(value) for value in all_channel_reverbs.values()]
    channels_participating_in_RSB = list(map(list, zip(*channel_reverbs)))

    n = 0
    for RSB in channels_participating_in_RSB:
        count = sum(1 for x in RSB if x >= 1)
        if count >= (len(all_channel_reverbs) * threshold):
            n += 1

    prop_NE_reverberating = n / len(channels_participating_in_RSB)

    return n, prop_NE_reverberating

'''
Network burst features
'''
def detect_networkReverb(network_sdf, fs=12500, min_ibi=0.2, network_burst_tolerance=0.8):

    min_ibi_ind = int(min_ibi * fs)

    burst_peaks, _ = find_peaks(network_sdf, height=8, distance=min_ibi_ind, prominence=5)
    if len(burst_peaks) >= 3:
        burst_heights = network_sdf[burst_peaks]
        ibi = np.diff(burst_peaks / fs)
        log_ibi = np.log(ibi)

        features = np.transpose([log_ibi, burst_heights[1:]])
        kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 250,
            "random_state": 0
        }

        kmeans = KMeans(n_clusters=2, **kmeans_kwargs)
        kmeans.fit(features)

        x = np.transpose(features)[0]
        y = np.transpose(features)[1]

        cluster_center_x = np.transpose(kmeans.cluster_centers_)[0, :]
        cluster_center_y = np.transpose(kmeans.cluster_centers_)[1, :]

        # Determine which cluster label belongs to prime peaks
        if (cluster_center_y[0] > cluster_center_y[1]) & (cluster_center_x[0] > cluster_center_y[1]):
            cluster_x_1 = x[kmeans.labels_ == 0]
            cluster_x_2 = x[kmeans.labels_ == 1]
            cluster_y_1 = y[kmeans.labels_ == 0]
            cluster_y_2 = y[kmeans.labels_ == 1]
            kmeans.labels_ = kmeans.labels_ + 1
        else:
            cluster_x_1 = x[kmeans.labels_ == 1]
            cluster_x_2 = x[kmeans.labels_ == 0]
            cluster_y_1 = y[kmeans.labels_ == 1]
            cluster_y_2 = y[kmeans.labels_ == 0]

        amax = min(cluster_y_1)
        rmax = 10**(min(cluster_x_1))
    else:
        rmax = 0.01
        amax = 1

    return rmax, amax

def networkBurstHist(network_sdf, nRandomSamples=10000, rmax=0.1, fs=12500):
    randomSample = random.sample(list(network_sdf), nRandomSamples)
    noise_threshold = np.median(randomSample)
    nb_threshold = noise_threshold + 8

    min_IBI = rmax
    min_duration = 0.01

    above_threshold = network_sdf >= nb_threshold

    # Find the indices where there are changes in the above_threshold state
    change_indices = np.where(np.diff(above_threshold))[0] + 1

    # Separate start and end indices
    starts = change_indices[above_threshold[change_indices - 1] == False]  # transition from False to True
    ends = change_indices[above_threshold[change_indices - 1] == True]  # transition from True to False

    # Handle cases where the series starts or ends above the threshold
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(network_sdf))

    # Combine starts and ends into a list of tuples
    peak_intervals = list(zip(starts / fs, ends / fs))

    # Merge close intervals
    i = 0
    while i < len(peak_intervals) - 1:
        # Check if the gap between current end and next start is less than thresholdY
        if peak_intervals[i + 1][0] - peak_intervals[i][1] < min_IBI:
            # Merge intervals
            peak_intervals[i] = (peak_intervals[i][0], peak_intervals[i + 1][1])
            del peak_intervals[i + 1]
        else:
            i += 1

    # Filter out short peaks
    nb_boundaries = [interval for interval in peak_intervals if (interval[1] - interval[0]) >= min_duration]

    return nb_boundaries


def countChannelReverberations(network_burst_start, network_burst_end, all_burst_data):
    all_channel_reverbs = {}
    mean_channel_reverbs = []
    median_channel_reverbs = []
    for c in range(len(all_burst_data)):
        channel_data = all_burst_data[c]
        middle_of_burst = []
        for b in channel_data:
            burst = channel_data[b]
            middle_of_burst.append(burst[int(len(burst) / 2)])
        channel_reverbs = []
        for n in range(len(network_burst_start)):
            start = network_burst_start[n]
            end = network_burst_end[n]
            channel_reverbs.append(sum((middle_of_burst >= start) & (middle_of_burst <= end)) - 1)
        all_channel_reverbs[c] = channel_reverbs
        mean_channel_reverbs.append(np.nanmean([[r for r in channel_reverbs if r > 0]]))
        median_channel_reverbs.append(np.nanmedian([[r for r in channel_reverbs if r > 0]]))

    return all_channel_reverbs, mean_channel_reverbs, median_channel_reverbs

def nRSBs(all_channel_reverbs, threshold=0.3):
    channel_reverbs = [list(value) for value in all_channel_reverbs.values()]
    channels_participating_in_RSB = list(map(list, zip(*channel_reverbs)))

    n = 0
    for RSB in channels_participating_in_RSB:
        count = sum(1 for x in RSB if x >= 1)
        if count >= (len(all_channel_reverbs) * threshold):
            n += 1

    prop_NE_reverberating = n / len(channels_participating_in_RSB)

    return n, prop_NE_reverberating

def calculate_networkBurstDuration(nb_boundaries):
    return [x[1]-x[0] for x in nb_boundaries], np.nanmean([x[1]-x[0] for x in nb_boundaries]), np.nanstd([x[1]-x[0] for x in nb_boundaries])

def calculate_networkIBI(nb_boundaries):
    networkIBI = [nb_boundaries[i+1][0] - nb_boundaries[i][1] for i in range(len(nb_boundaries)-1)]
    return networkIBI, np.nanmean(networkIBI), np.nanstd(networkIBI)

def is_outside_intervals(spike, intervals):
    for (start, end) in intervals:
        if start <= spike <= end:
            return False
    return True

def calculate_randomSpiking(raster, nb_boundaries):
    spiketimes = list(chain.from_iterable(raster))
    count = sum(1 for spike in spiketimes if is_outside_intervals(spike, nb_boundaries))
    return count

'''
Functional connectivity features
'''
def STTC(spiketrain_1, spiketrain_2, duration=600, dt=0.005):
    '''
    Adapted from: https://github.com/NeuralEnsemble/elephant/blob/master/elephant/spike_train_correlation.py
    1/2((PA-TB)/(1-PA*TB) + (PB-TA)/(1-PB*TA)
    '''
    def run_P(spiketrain_1, spiketrain_2):
        N2 = len(spiketrain_2)
        ind = np.searchsorted(spiketrain_2, spiketrain_1)
        ind[ind == N2] = N2 - 1
        close_left = np.abs(spiketrain_2[ind - 1] - spiketrain_1) <= dt
        close_right = np.abs(spiketrain_2[ind] - spiketrain_1) <= dt
        close = close_left + close_right
        return np.count_nonzero(close)

    def run_T(spiketrain):
        N = len(spiketrain)
        time_A = 2 * N * dt  # maximum possible time
        if N == 1:  # for only a single spike in the train
            if spiketrain[0] - 0 < dt:
                time_A += - dt + spiketrain[0] - 0
            elif spiketrain[0] + dt > duration:
                time_A += - dt - spiketrain[0] + duration
        else:  # if more than a single spike in the train
            diff = np.diff(spiketrain)
            idx = np.where(diff < 2 * dt)[0]
            time_A += - 2 * dt * len(idx) + diff[idx].sum()
            if (spiketrain[0] - 0) < dt:
                time_A += spiketrain[0] - dt - 0
            if (duration - spiketrain[N - 1]) < dt:
                time_A += - spiketrain[-1] - dt + duration
        T = time_A / (duration - 0)
        return T  # enforce simplification, strip units

    N1 = len(spiketrain_1)
    N2 = len(spiketrain_2)

    if N1 == 0 or N2 == 0:
        index = np.nan
    else:
        TA = run_T(spiketrain_1)
        TB = run_T(spiketrain_2)
        PA = run_P(spiketrain_1, spiketrain_2)
        PA = PA / N1
        PB = run_P(spiketrain_2, spiketrain_1)
        PB = PB / N2
        if PA * TB == 1:
            if PB * TA == 1:
                index = 1.
            else:
                index = 0.5 + 0.5 * (PB - TA) / (1 - PB * TA)
        elif PB * TA == 1:
            index = 0.5 + 0.5 * (PA - TB) / (1 - PA * TB)
        else:
            index = 0.5 * (PA - TB) / (1 - PA * TB) + 0.5 * (PB - TA) / (
                1 - PB * TA)
    return index

def calculate_sttc(well, duration, dt=0.005):
    sttc_m = np.ones((len(well), len(well))) * np.nan
    for i, ch1 in enumerate(well):
        for j, ch2 in enumerate(well):
            try:
                sttc_m[i][j] = STTC(ch1, ch2, duration, dt)
            except:
                print(f"Ran into error when comparing {i} by {j}")
    return sttc_m


'''
PLOTTING ASSISTANCE
'''

def plotColorBurstRaster(all_burst_data):
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10,5))
    # Define colors for bursts
    colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    # Iterate over rows
    for i, row_key in enumerate(sorted(all_burst_data.keys())):
        row_data = all_burst_data[row_key]

        # Iterate over bursts in the row
        for burst_key in sorted(row_data.keys()):
            burst_spike_times = row_data[burst_key]
            burst_color = next(colors)

            # Plot spike times
            ax.scatter(burst_spike_times, np.ones_like(burst_spike_times) * i, color=burst_color, label=f'Row {row_key}, Burst {burst_key}', s=3)

    # Set labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Channel Number')
    # Show plot
    plt.show()

def plotNetworkBurstBoundaries(raster, time, network_sdf, nb_boundaries):
    plt.plot(time, network_sdf/16)
    plt.eventplot(raster)
    for b in nb_boundaries:
        plt.axvline(b[0], color='green')
        plt.axvline(b[1], color='red')

def plotChannelReverb_networkBurst(all_channel_reverbs):
    channel_reverbs = [list(value) for value in all_channel_reverbs.values()]
    plt.imshow(channel_reverbs)
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel("Network event")
    plt.ylabel("Channel")