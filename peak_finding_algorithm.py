# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 21:23:45 2021

@author: 184277J
"""


from scipy import signal
import numpy as np
import matplotlib.pyplot as plt



def my_sqrt(val):
    return np.sign(val)*np.sqrt(np.abs(val))


offset = 0 # in channels. Is there an off-by-one error in the peakfinding things?

data_x, data_y , _ = np.loadtxt("corr_KH_Al2S3_61_ss001.xye", unpack = True)

width_min = 1
width_max = 31


# set up all the defaults and things you need
vector = data_y
widths = np.arange(width_min, width_max)
wavelet = signal.ricker
max_distances_divisor = 1 #default = 4 # At each row, a ridge line is only connected if the relative max at row[n] is within ``max_distances[n]`` from the relative max at ``row[n+1]``.
gap_thresh = 2 #A ridge line is discontinued if there are more than `gap_thresh` points without connecting a new relative maximum.
min_length = 3 # Minimum length a ridge line needs to be acceptable.
min_snr = 3
noise_perc = 20
window_size = None #Size of window to use to calculate noise floor.

max_distances = widths / max_distances_divisor #None
# alter them accordingly
if gap_thresh is None:
    gap_thresh = np.ceil(widths[0])
if max_distances is None:
    max_distances = widths / 4.0


# do the thing
cwt = signal.cwt(vector, wavelet, widths, window_size=window_size)
ridge_lines = signal._peak_finding._identify_ridge_lines(cwt, max_distances, gap_thresh)

filtered =signal._peak_finding._filter_ridge_lines(cwt, ridge_lines, min_length=min_length,
                                                   window_size=window_size, min_snr=min_snr,
                                                   noise_perc=noise_perc)

max_locs = np.asarray([x[1][0] for x in filtered])

# peak positions
max_locs.sort()
pks = np.take(data_x, max_locs-offset)




fig, (ax1, ax2) = plt.subplots(2, sharex = True, gridspec_kw={'height_ratios': [1, 2]})
fig.suptitle('Vertically stacked subplots')
#ax1.plot(x, y)
#ax2.plot(x, -y)


for i in range(len(ridge_lines)):
    ax1.scatter(np.take(data_x, ridge_lines[i][1]-offset), width_min-np.take(widths,ridge_lines[i][0])+width_max, marker = "|")

for i in range(len(filtered)):
    ax1.scatter(np.take(data_x, filtered[i][1]-offset), width_min-np.take(widths,filtered[i][0])+width_max, marker = "_", color = "k", linewidth = 0.5)


#for i in range(len(filtered)):
#    plt.scatter(filtered[i][1]/100 - 1, -filtered[i][0]+30, marker = "|")


ax1.imshow(my_sqrt(cwt), extent=[data_x[0], data_x[-1], width_min, width_max], cmap='RdBu', aspect='auto',
           vmax=my_sqrt(abs(cwt)).max(), vmin=-my_sqrt(abs(cwt)).max())

ax1.invert_yaxis()


for pk in pks:
    ax2.axvline(x=pk, c = "r", lw = 1, ls = ":")

ax2.plot(data_x, my_sqrt(data_y), color = "#0000ff")   #, marker = "+"
#plt.plot(t,sig*4 + 10)



plt.show()














