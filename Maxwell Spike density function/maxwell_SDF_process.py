from pathlib import Path
import pandas as pd
import numpy as np
import math
from scipy.stats import norm
from scipy.signal import convolve
from scipy.signal import resample
import sys
import pickle
import time
import multiprocessing as mp
from h5py import File
from tqdm import tqdm

def extract_spikes_from_h5(filename):
    '''
    :param filename: Accepts hdf5 file
    :return: channel: all channels (associated with a single spike time; see below)
    :return: time: all spike times (associated with a single channel; see above)
    :return: raster: organized spike times into a list of channels with their respective spike times
    '''
    try:
        f = File(filename, "r")
        recordings = f["data_store"]["data0000"]
        spikes = np.array(recordings["spikes"])

        channel = spikes["channel"]
        time = (spikes["frameno"] - (min(spikes["frameno"]))) / 20000  # If I remember correctly, frameno does not start at 0 but rather datetime of when the recording took place
        duration = recordings["stop_time"][0] - recordings["start_time"][0] # This doesn't get used

        # This part just organizes the spiking times based on the channel they belong to.
        # Produces a list of list, where each list contains all the spikes times belonging to a specific channel.
        channel_times = {}
        for c, t in zip(channel, time):
            if c not in channel_times:
                channel_times[c] = []
            channel_times[c].append(t)
        raster = [channel_times[channel] for channel in sorted(channel_times)]
        return channel, time, raster

    except:
        print("Path does not contain an h5 file")
        return [], [], []

def generate_gaussian_kernel(sigma=0.05, fs=20000):
    '''
    :param sigma: Width of kernel (s)
    :param fs: Sampling frequency in Hz
    :return: Gaussian kernel
    '''
    edges = np.arange(-3 * sigma, 3 * sigma, 1 / fs)
    kernel = norm.pdf(edges, 0, sigma)
    kernel = kernel / max(kernel)
    return kernel

def generate_spikematrix(spiketrain, fs, duration):
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

def generate_sdf(spikematrix, kernel):
    '''
    :param spikematrix: Binary matrix containing spikes
    :param gaussian_kernel: Gaussian kernel to be convolved with spikematrix
    :return: sdf: Continuous timeseries representing probability distribution of activity
    '''
    sdf_tmp = convolve(spikematrix, kernel)
    # Convolution will change the size of the recording so lets maintain the original spike train length
    sdf = sdf_tmp[int((len(sdf_tmp)-len(spikematrix))/2):int(len(sdf_tmp)-((len(sdf_tmp)-len(spikematrix))/2))]
    sdf = sdf/max(kernel) # Make height of sdf related to firing rate
    return sdf


'''
THIS IS THE PROCESS THAT GETS RUN
'''
kernel = generate_gaussian_kernel(sigma=0.01, fs=20000)
def compute_features(filename):
    print(f"Currently in : {filename}")
    fs = 20000
    RESAMPLE_FACTOR = 25 # Just to make the file size smaller. Don't need to include all the data. Change to 1 to retain all information

    # Collect some useful metadata
    filename = str(filename)
    recording_folder = filename.split("/")[-3]
    date = filename.split("/")[-5]
    chipid = filename.split("/")[-4]

    try:
        channel, time, raster = extract_spikes_from_h5(filename)
        duration = 300 # Duration of the recording
        network_sdf = np.zeros(duration*fs) # Initialize the store of the network spike density function
        for n, ch in tqdm(enumerate(raster), total=len(raster)): # Progress bar for loop. Uses tqdm library
        #for n, ch in enumerate(raster):
            spiketrain = raster[n]
            channel_spikematrix = generate_spikematrix(spiketrain, fs=fs, duration=duration)
            sdf_tmp = generate_sdf(channel_spikematrix, kernel)

            # This adds the contribution of each channel to the growing network spike density function.
            # Original attempt would store each sdf_tmp in memory and then average at the end, that used 1000 times more memory
            network_sdf += sdf_tmp

        ds_network_sdf = resample(network_sdf, int(len(network_sdf) / RESAMPLE_FACTOR))
        return [filename, recording_folder, date, chipid, channel, time, raster, ds_network_sdf]
    except:
        return [filename, recording_folder, date, chipid, [], [], [[]], [], []]