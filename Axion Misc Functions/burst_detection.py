import numpy as np
from scipy.signal import find_peaks
from burst_clusters import detect_reverb
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.signal import resample
import time
import math
from itertools import chain


def maxInterval(spiketrain, max_begin_ISI=0.17, max_end_ISI=0.3, min_IBI=0.2, min_burst_duration=0.01,
                min_spikes_in_burst=3):
    allBurstData = {}

    '''
    Phase 1 - Burst Detection
    Here a burst is defined as starting when two consecutive spikes have an
    ISI less than max_begin_ISI apart. The end of the burst is given when two
    spikes have an ISI greater than max_end_ISI.
    Find ISIs closer than max_begin_ISI and end with max_end_ISI.
    The last spike of the previous burst will be used to calculate the IBI.
    For the first burst, there is no previous IBI.
    '''
    inBurst = False
    burstNum = 0
    currentBurst = []
    for n in range(1, len(spiketrain)):
        ISI = spiketrain[n] - spiketrain[n - 1]
        if inBurst:
            if ISI > max_end_ISI:  # end the burst
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                allBurstData[burstNum] = currentBurst
                currentBurst = []
                burstNum += 1
                inBurst = False
            elif (ISI < max_end_ISI) & (n == len(spiketrain) - 1):
                currentBurst = np.append(currentBurst, spiketrain[n])
                allBurstData[burstNum] = currentBurst
                burstNum += 1
            else:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
        else:
            if ISI < max_begin_ISI:
                currentBurst = np.append(currentBurst, spiketrain[n - 1])
                inBurst = True
    # Calculate IBIs
    IBI = []
    for b in range(1, burstNum):
        prevBurstEnd = allBurstData[b - 1][-1]
        currBurstBeg = allBurstData[b][0]
        IBI = np.append(IBI, (currBurstBeg - prevBurstEnd))

    '''
    Phase 2 - Merging of Bursts
    Here we see if any pair of bursts have an IBI less than min_IBI; if so,
    we then merge the bursts. We specifically need to check when say three
    bursts are merged into one.
    '''
    tmp = allBurstData.copy()
    allBurstData = {}
    burstNum = 0
    b = 1
    while b < len(tmp):
        prevBurst = tmp[b - 1]
        currBurst = tmp[b]
        if IBI[b - 1] <= min_IBI:
            prevBurst = np.append(prevBurst, currBurst)
            b += 1
        allBurstData[burstNum] = prevBurst
        b+=1
        burstNum += 1
    if burstNum >= 2:
        allBurstData[burstNum] = currBurst

    '''
    Phase 3 - Quality Control
    Remove small bursts less than min_bursts_duration or having too few
    spikes less than min_spikes_in_bursts. In this phase we have the
    possibility of deleting all spikes.
    '''
    tooShort = 0
    tmp = allBurstData
    allBurstData = {}
    burstNum = 0
    if len(tmp) > 1:
        for b in range(len(tmp)):
            currBurst = tmp[b]
            if len(currBurst) <= min_spikes_in_burst:
                tooShort +=1
            elif currBurst[-1] - currBurst[0] <= min_burst_duration:
                tooShort += 1
            else:
                allBurstData[burstNum] = currBurst
                burstNum += 1
    '''
    plt.figure()
    plt.eventplot(spiketrain)
    for b in allBurstData:
        burst = allBurstData[b]
        plt.axvline(burst[0], color='green')
        plt.axvline(burst[-1], color='red')
    '''
    return allBurstData, tooShort

'''
Network Burst Detection from MaxInterval-detected bursts
Start of network burst is when at least 35% of the active channels 
come online within 500ms of each other.
End of network burst is when at least 35% of the active channels 
turn off.
'''
def calculateBurstWindow(all_burst_data):
    all_burst_start = {}
    all_burst_end = {}
    for channel_ind in all_burst_data.keys():
        channel_bursts = all_burst_data[channel_ind]
        burst_start = []
        burst_end = []
        for burst_ind in channel_bursts.keys():
            burst_start.append(channel_bursts[burst_ind][0])
            burst_end.append(channel_bursts[burst_ind][-1])
        all_burst_start[channel_ind] = burst_start
        all_burst_end[channel_ind] = burst_end
    return all_burst_start, all_burst_end

def networkBurstDetection_old(all_burst_data, min_electrodes=0.35, window=0.1):
    all_burst_start, all_burst_end = calculateBurstWindow(all_burst_data, window)
    network_burst_start, network_burst_end = [], []

    all_burst_start = [burst_start for channel_burst_start in all_burst_start.values() for burst_start in channel_burst_start]
    all_burst_end = [burst_start for channel_burst_start in all_burst_end.values() for burst_start in channel_burst_start]

    max_duration = math.ceil(max(all_burst_end)/100) * 100
    start_counts, common_burst_starts = np.histogram(all_burst_start, int(max_duration/window))
    common_burst_starts = common_burst_starts[:-1]
    end_counts, common_burst_ends = np.histogram(all_burst_end, int(max_duration/window/2))
    common_burst_ends = common_burst_ends[:-1]

    plt.plot(common_burst_starts, start_counts, color='green')
    plt.plot(common_burst_ends, end_counts, color='red')


    network_burst_start = common_burst_starts[start_counts >= 5]
    network_burst_end = common_burst_ends[end_counts >= 4]
    print(f"Length of start: {len(network_burst_start)} --- Length of end: {len(network_burst_end)}")
    pass


def find_histogram_edges(network_counts, start_threshold=2, end_threshold=1):
    edges = []
    exceeding_threshold = False
    fr = []
    for i, count in enumerate(network_counts):
        if exceeding_threshold and count <= end_threshold:
            edges.append(i)
            exceeding_threshold = False
            fr.append(tmp)
        elif not exceeding_threshold and count >= start_threshold:
            edges.append(i)
            exceeding_threshold = True
            tmp = 0
        if exceeding_threshold and network_counts[i] > tmp:
            tmp = network_counts[i]

    return edges, fr

def fix_burst_borders_length_mismatch(burst_start, burst_end):
    def check_increasing(x, y):
        if len(x) != len(y):
            return False, np.nan
        for i in range(len(x)):
            if x[i] >= y[i]:
                return False, i
        return True, -1

    def first_negative(d):
        for i, num in enumerate(d):
            if num < 0:
                return i
        return -1
    # Are the networks
    isIncreasing, ind = check_increasing(burst_start, burst_end)

    if isIncreasing == False:
        print(f"Network Burst length mismatch")
        if len(burst_end) > len(burst_start):
            # Check if burst_start is too long from the front or from the end
            diff = np.array(burst_end)[:-1]-np.array(burst_start)
            burst_end = list(burst_end)
            burst_start = list(burst_start)
            burst_end.pop(first_negative(diff))
            return burst_start, burst_end
        elif len(burst_start) > len(burst_end):
            # Check if burst_end is too long from the front or from the end
            diff = np.array(burst_start)[1:] - np.array(burst_end)
            burst_end = list(burst_end)
            burst_start = list(burst_start)
            burst_start.pop(first_negative(diff))
            return burst_start,burst_end
        else:
            print(f"Ran into an increasing error that could not be fixed.")
            return burst_start,burst_end

    return burst_start, burst_end

def histNetworkBurstDetection(well_raster, resolution=0.1, start_threshold=2, end_threshold=1):
    all_spikes = []
    for ch in well_raster:
        all_spikes = np.append(all_spikes, ch)

    max_duration = math.ceil(max(all_spikes)/100) * 100
    network_counts, network_bins = np.histogram(all_spikes, int(max_duration/resolution))
    network_counts = network_counts/len(well_raster)

    edges, network_burst_fr = find_histogram_edges(network_counts, start_threshold, end_threshold)

    network_burst_start_ind = [edges[i] for i in range(len(edges)) if i%2==0]
    network_burst_start = network_bins[network_burst_start_ind]
    network_burst_end_ind = [edges[i] for i in range(len(edges)) if i%2==1]
    network_burst_end = network_bins[network_burst_end_ind]

    network_burst_start, network_burst_end = fix_burst_borders_length_mismatch(network_burst_start, network_burst_end)

    return network_burst_start, network_burst_end, network_burst_fr

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

def countNonNetworkBurstBursts(network_burst_start, network_burst_end, all_burst_data):
    channel_nonNBBursts = []
    for c in range(len(all_burst_data)):
        count = 0
        channel_data = all_burst_data[c]
        middle_of_burst = []
        for b in channel_data:
            burst = channel_data[b]
            middle_of_burst.append(burst[int(len(burst)/2)])
        for b in middle_of_burst:
            is_overlapping = False
            for n in range(len(network_burst_start)):
                if network_burst_start[n] <= b <= network_burst_end[n]:
                    is_overlapping = True
                    break
            if not is_overlapping:
                count+=1
        channel_nonNBBursts.append(count)
    return channel_nonNBBursts, np.nanmean(channel_nonNBBursts), np.nanmedian(channel_nonNBBursts)

def countNetworkReverberations(network_burst_start, network_burst_end, highRes_network_burst_start, highRes_network_burst_end):
    nReverbs = []
    for b in range(len(network_burst_start)):
        r = -1
        for m in range(len(highRes_network_burst_start)):
            if ((highRes_network_burst_start[m] >= network_burst_start[b]) and (highRes_network_burst_start[m] <= network_burst_end[b])) \
                    or ((highRes_network_burst_end[m] <= network_burst_end[b]) and (highRes_network_burst_end[m] > network_burst_start[b])):
                r+=1
        nReverbs.append(r)
    for b in range(len(nReverbs)):
        if nReverbs[b] == -1:
            nReverbs[b] = 0
    return nReverbs

def sdfNetworkBurstDetection(network_sdf, burst_tolerance=0.3, fs=12500):
    def calculate_logrmax(burst_peaks):
        counts, binEdges = np.histogram(np.log(np.diff(burst_peaks)), 50)
        if len(np.where(counts == 0)[0]) > 0:
            tmp = np.where(counts == 0)[0][0]
            rmax = binEdges[tmp]
        return 10**rmax

    def calculate_rmax(burst_peaks):
        counts, binEdges = np.histogram(np.diff(burst_peaks), 50)
        if len(np.where(counts == 0)[0]) > 0:
            tmp = np.where(counts == 0)[0][0]
            rmax = binEdges[tmp]
        return rmax

    def detect_burst_borders(burst_peaks, network_sdf, threshold=0.3, downsample=100):
        ds_network_sdf = resample(network_sdf, int(len(network_sdf) / downsample))
        dv_ds = np.diff(ds_network_sdf)/max(np.diff(ds_network_sdf))
        burst_start,_ = find_peaks(dv_ds, prominence=threshold)
        burst_end, _ = find_peaks(-dv_ds, prominence=threshold)
        burst_start *= downsample
        burst_end *= downsample
        return burst_start/12500, burst_end/12500

    '''    
    plt.plot(ds_network_sdf/max(ds_network_sdf))
    plt.plot(dv_ds)
    for start in burst_start:
        plt.axvline(start, color='green')
    for end in burst_end:
        plt.axvline(end, color='red')
    '''
    def fix_burst_borders_length_mismatch(burst_start, burst_end):
        def check_increasing(x, y):
            if len(x) != len(y):
                return False, np.nan
            for i in range(len(x)):
                if x[i] >= y[i]:
                    return False, i
            return True, -1

        def first_negative(d):
            for i, num in enumerate(d):
                if num < 0:
                    return i
            return -1
        # Are the networks
        isIncreasing, ind = check_increasing(burst_start, burst_end)

        if isIncreasing == False:
            if len(burst_end) > len(burst_start):
                # Check if burst_start is too long from the front or from the end
                diff = np.array(burst_end)[:-1]-np.array(burst_start)
                burst_end = list(burst_end)
                burst_start = list(burst_start)
                burst_end.pop(first_negative(diff))
                return burst_start, burst_end
            elif len(burst_start) > len(burst_end):
                # Check if burst_end is too long from the front or from the end
                diff = np.array(burst_start)[1:] - np.array(burst_end)
                burst_end = list(burst_end)
                burst_start = list(burst_start)
                burst_start.pop(first_negative(diff))
                return burst_start,burst_end
            else:
                print(f"Ran into an increasing error that could not be fixed.")
                return burst_start,burst_end

        return burst_start, burst_end

    def merge_bursts(burst_start, burst_end, rmax):
        ne_start, ne_end, num_reverbs = [], [], []
        r = 0
        in_super_burst = False
        if (len(burst_start) != len(burst_end)):
            print("Border length mismatch. Attempting to fix.")
            burst_start, burst_end = fix_burst_borders_length_mismatch(burst_start,burst_end)
        if (len(burst_start) == len(burst_end)):
            for b in range(0, len(burst_start) - 1):
                if in_super_burst == False:
                    ne_start.append(burst_start[b])
                if (burst_start[b + 1] - burst_end[b]) >= rmax:
                    ne_end.append(burst_end[b])
                    num_reverbs.append(r)
                    r = 0
                    in_super_burst = False
                else:
                    in_super_burst = True
                    r += 1
            if (burst_start[b+1] - burst_end[b]) <= rmax:
                ne_end.append(burst_end[b+1])
                num_reverbs.append(r)
            else:
                ne_start.append(burst_start[b+1])
                ne_end.append(burst_end[b+1])
        else:
            print("Could not fix length mismatch.")
        return ne_start, ne_end, num_reverbs

    '''
    plt.plot(ds_network_sdf / max(ds_network_sdf))
    plt.plot(dv_ds)
    for start in ne_start:
        plt.axvline(start, color='green')
    for end in ne_end:
        plt.axvline(end, color='red')
    '''

    start = time.time()
    maxFR = max(network_sdf)
    burst_peaks, properties = find_peaks(network_sdf, height=maxFR*0.2, prominence=maxFR*0.1, rel_height=0.5, width=5, distance=10)
    burst_peaks = burst_peaks/fs
    burst_start, burst_end = detect_burst_borders(burst_peaks, network_sdf, threshold=burst_tolerance, downsample=100)
    logrmax = calculate_logrmax(burst_peaks)
    me_start, me_end, me_num_reverbs = merge_bursts(burst_start, burst_end, logrmax)
    rmax = calculate_rmax(burst_peaks)
    ne_start, ne_end, ne_num_reverbs = merge_bursts(burst_start, burst_end, rmax)
    end = time.time()
    print(f"Network and oscillatory bursting activity took {end-start} seconds.")

    return burst_peaks, logrmax, rmax, burst_start, burst_end, me_start, me_end, me_num_reverbs, ne_start, ne_end, ne_num_reverbs
