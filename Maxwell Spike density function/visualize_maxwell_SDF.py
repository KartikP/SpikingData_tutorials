import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import numpy as np

savename = "Maxwell_SDF"
filehandler = open(savename, 'rb')
df = pickle.load(filehandler)
print("File saved and loaded successfully.")

duration = 300 # This only affects the SDF time axis

'''
Visualize raster plot and spike density function overtop
'''
# Let's just look at one of the files
i = 1

raster = df.iloc[i]["raster"]
sdf = df.iloc[i]["ds_network_sdf"]

time = np.arange(0, duration, duration/len(sdf)) # duration/len(sdf) calculates the 1/sampling frequency

fig = plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[1, 0])

ax0.plot(time, sdf)
ax0.set_ylabel("Network Firing Rate (Hz)")

ax1.eventplot(raster)
ax1.set_ylabel("Channel number")
ax1.set_xlabel("Time (s)")

plt.show()

'''
Visualize the most active network burst based on the maximum peak of the SDF
This is a bad way to do it because it doesn't give you flexibility in terms of
when the network burst starts and ends, but I didn't have time to adaptively determine
thresholds.
'''
max_burst_ind = np.argmax(sdf)
max_burst_time = (max_burst_ind*(duration/len(sdf)))

ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 1])

ax2.plot(time,sdf)
ax2.set_xlim([max_burst_time-1, max_burst_time+2])
ax2.set_ylabel("Network Firing Rate (Hz)")

ax3.eventplot(raster, linewidths=2)
ax3.set_xlim([max_burst_time-1, max_burst_time+2])
ax3.set_ylabel("Channel number")
ax3.set_xlabel("Time (s)")

