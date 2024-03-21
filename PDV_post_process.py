
"""
@author: Aidanklemmer
University of Nevada, Reno
Aidanklemmer@outlook.com
3/21/24
"""

### PDV post-processing 
#
#
#
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.stats as stats
from scipy.signal import savgol_filter

#---
### PDV class
#
# each shot has many objects within the class
#
class PDVshot:
    
    def __init__(self, shot_num, tau): # initialize class for each shot number
    
        # initialize class object
        self.shot_num = shot_num # shot number is the "name"
        self.tau = tau # STFT window duration (tau)
        
        # region of interest for physics
        self.t_start = 45
        self.t_end = 72
        self.idx_start = []
        self.idx_end = []
        
        # digitizer speed (sample rate)
        # ED1 campaign used 25 GHz (0.04 ns, 40 ps, point spacing), defined as "alpha"
        self.alpha = 40e-12 # 40 ps
        
        # storage for velocity-time history data
        self.velocity_data = []
        self.velocity_data_filtered = []
        self.time_data = []
        
        # storage for acceleration (dv/dt) data
        self.acceleration_data = []
        
        # storage for jerk (da/dt) data 
        self.jerk_data = []
        
        # ---
        ## physical variables
        # ---
        # acceleration
        self.val_max_accel = []
        self.val_min_accel = []
        self.time_max_accel = []
        self.time_min_accel = []
        
        # jerk
        self.val_max_jerk = []
        self.val_min_jerk = []
        self.time_max_jerk = []
        self.time_min_jerk = []
        
        # extrema 
        self.extrema_accel = []
        self.extrema_jerk = []
        
        # compression 
        self.max_neg_vel = []
        self.time_max_neg_vel = []
        self.compression_disp = [] # displacement from compression (integral of velocity from -20 ns until the v = 0)
        self.comp_t_stat = [] # compression t-test t-stat
        self.comp_p_val = [] # compression t-test p-value
        self.time_postive_disp = [] # time of positive (outward) displacement and "motion" 
        
        # times of interest
        self.time_vel_0 = [] # time of v = 0 m/s
        self.time_vel_2 = [] # time of v = 2 m/s
        self.time_vel_4 = [] # time of v = 4 m/s
    
        # time between extrema of jerk (duration of melt)
        self.duration_melt = []
        
        # baseline (reference) values
        self.baseline_vel = []
        self.baseline_vel_avg = []
        self.baseline_vel_stdev = []
        self.baseline_vel_rms = []
        self.baseline_accel_rms = []
        self.baseline_jerk_rms = [] 
        
        # jerk signal to noise ratio
        self.jerk_SNR = [] 
#---    
    
#---  
    ## class methods 
    #
    # load velocity history
    def add_vel_hist(self, time, vel):
        
        self.velocity_data.append(vel)
        self.time_data.append(time)
    
    # moving average 
    def smooth_data(self, window_size, num_iter):
        
        if num_iter == 1:
            smoothed_vel = np.convolve(self.velocity_data[0], np.ones(window_size)/window_size, mode='same')
            self.velocity_data_filtered.append(smoothed_vel)
        elif num_iter == 2:
            smoothed_vel1 = np.convolve(self.velocity_data[0], np.ones(window_size)/window_size, mode='same')
            smoothed_vel2 = np.convolve(smoothed_vel1, np.ones(window_size)/window_size, mode='same')
            self.velocity_data_filtered.append(smoothed_vel2)
        else:
            print('Invalid moving average filter iteration number')
            sys.exit(1) # raise system exception and code to exit
            
    # Savitzky-Golay (SG) derivatives and filters
    def calc_SG_derivatives(self, window_size):
        
        order = 3 # SG filter polynomial order
        smoothed_vel_SG = savgol_filter(self.velocity_data_filtered, window_size, order, delta=self.alpha)
        acceleration = savgol_filter(self.velocity_data_filtered, window_size, order, deriv=1, delta=self.alpha)
        jerk = sc.signal.savgol_filter(self.velocity_data_filtered, window_size, order, deriv=2, delta=self.alpha)
        acceleration = acceleration * 1e-9
        jerk = jerk * 1e-18
        self.velocity_data_filtered.clear()
        self.velocity_data_filtered.append(smoothed_vel_SG[0])
        self.acceleration_data.append(acceleration[0])
        self.jerk_data.append(jerk[0])
       
    def calc_baseline_noise(self):
        
        # baseline 
        tbase_start = -360
        tbase_end = 0
        base_start = np.where(self.time_data[0] >= tbase_start)[0][0]
        base_end = np.where(self.time_data[0] <= tbase_end)[0][-1]
        self.baseline_vel = self.velocity_data_filtered[0][base_start:base_end]
        self.baseline_vel_avg = np.mean(self.baseline_vel[0])
        self.baseline_vel_stdev = np.std(self.baseline_vel[0])
        self.baseline_vel_rms = np.sqrt((np.mean(self.baseline_vel[0]**2)))
    
    def calc_compression_disp(self):   
        
        # compression displacement
        tcomp_start = 20
        vel_0_idx = np.where(self.velocity_data[0] <= 0)[0][-1]
        comp_start_idx = np.where(self.time_data[0] >= tcomp_start)[0][0]
        comp_reg = self.velocity_data[0][comp_start_idx:vel_0_idx]
        self.comp_disp = - sc.integrate.simpson(comp_reg, dx=self.alpha) * 1e9 # make postive and convert to nm
    
    def calc_compression_t_test(self):
        
        # find the indices that correspond to the time range
        # time domain (x1, and x2 data) are the same for both first and second derivative
        self.idx_start = np.where(self.time_data[0] >= self.t_start)[0][0]
        self.idx_end = np.where(self.time_data[0] <= self.t_end)[0][-1]
        
        # calculate time of max compression
        comp_max_vel = np.min(self.velocity_data[0][self.idx_start:self.idx_end]) # find max neg vel, this is the min vel value in the range
        self.max_neg_vel = comp_max_vel
        ind_comp_max = np.where(self.velocity_data[0] <= comp_max_vel)
        max_neg_vel_idx = ind_comp_max[0][0]  # index of the center value
        self.time_max_neg_vel = self.time_data[0][max_neg_vel_idx]
        stat_shift = 0  # 3 ns shift earlier in time, this is an arbitrary choice
        unique_pt_spacing_func_tau = int(((self.tau *1e-9) / self.alpha) + 1)
        offsets = [unique_pt_spacing_func_tau + stat_shift,  0 + stat_shift, unique_pt_spacing_func_tau + stat_shift] # define the offsets/spacing for the 3 pt comp region population
        vel_3pt_comp_pop = self.velocity_data[0][[max_neg_vel_idx + offset for offset in offsets]] # calculate the 3 pt comp population
        vel_3pt_comp_pop = [round(val, 2) for val in vel_3pt_comp_pop] # round
        time_center_3pt_comp_pop = self.time_data[0][max_neg_vel_idx] # calculate the center time of the 3 pt comp population   
        # calculate t-test statistics 
        # Welch t-test
        self.comp_t_stat, self.comp_p_val = stats.ttest_ind(vel_3pt_comp_pop, np.zeros(3), equal_var=False)  # perform T-test
 
    def calc_extrema(self):
        
        #---
        ## perform peakfinding and calculate extrema 
        #---
        # find zero crossings
        #zero_crossings_accel = np.where(np.diff(np.sign(self.acceleration_data)))[0] # acceleration
        #zero_crossings_jerk = np.where(np.diff(np.sign(self.jerk_data)))[0] # jerk
        
        # find extrema using scipy.signal.find_peaks
        # acceleration
        peaks_accel, _ = find_peaks(self.acceleration_data[0])
        valleys_accel, _ = find_peaks(-np.array(self.acceleration_data[0]))
        self.extrema_accel = np.sort(np.concatenate((peaks_accel, valleys_accel))) 
        
        # find extrema using scipy.signal.find_peaks
        # jerk
        peaks_jerk, _ = find_peaks(self.jerk_data[0])
        valleys_jerk, _ = find_peaks(-np.array(self.jerk_data[0]))
        self.extrema_jerk = np.sort(np.concatenate((peaks_jerk, valleys_jerk)))
    
    def calc_peaks_derivatives(self):
        
        # calculate zero crossing and extrema of acceleration
        # take only data that falls within the time range
        #zero_crossings_range_accel = zero_crossings_accel[(zero_crossings_accel >= idx_start) & (zero_crossings_accel <= idx_end)] # not currently used
        extrema_range_accel = self.extrema_accel[(self.extrema_accel >= self.idx_start) & (self.extrema_accel <= self.idx_end)]
        
        # calculate zero crossing and extrema of jerk
        # take only data that falls within the time range
        #zero_crossings_range_jerk = zero_crossings_jerk[(zero_crossings_jerk >= idx_start) & (zero_crossings_jerk <= idx_end)]
        extrema_range_jerk = self.extrema_jerk[(self.extrema_jerk >= self.idx_start) & (self.extrema_jerk <= self.idx_end)]
        
        # find max values of acceleration
        max_peak_time_accel = self.time_data[0][extrema_range_accel[np.argmax(self.acceleration_data[0][extrema_range_accel])]]
        # find max and min values of jerk
        max_peak_time_jerk = self.time_data[0][extrema_range_jerk[np.argmax(self.jerk_data[0][extrema_range_jerk])]]
        min_peak_time_jerk = self.time_data[0][extrema_range_jerk[np.argmin(self.jerk_data[0][extrema_range_jerk])]]
        
        # find duration of time between extrema of jerk
        self.duration_melt = min_peak_time_jerk - max_peak_time_jerk
        
        # index local minimum of acceleration after peak acceleration
        min_accel_after_max_idx = np.where(np.array(self.time_data[0])[extrema_range_accel] > max_peak_time_accel)
        
        # find max acceleration (local extrema)
        self.val_max_accel = self.acceleration_data[0][extrema_range_accel[np.argmax(self.acceleration_data[0][extrema_range_accel])]]
        # find min acceleration (local extrema)
        self.val_min_accel = np.array(self.acceleration_data[0])[extrema_range_accel][min_accel_after_max_idx][0]
        # time of max acceleration (local extrema)
        self.time_max_accel = max_peak_time_accel
        # time of min acceleration (local extrema)
        self.time_min_accel = np.array(self.time_data[0])[extrema_range_accel][min_accel_after_max_idx][0]
        # find max jerk (local extrema)
        self.val_max_jerk = self.jerk_data[0][extrema_range_jerk[np.argmax(self.jerk_data[0][extrema_range_jerk])]]
        # find min jerk (local extrema)
        self.val_min_jerk = self.jerk_data[0][extrema_range_jerk[np.argmin(self.jerk_data[0][extrema_range_jerk])]]
        # time of max jerk (local extrema)
        self.time_max_jerk = max_peak_time_jerk
        # time of min jerk (local extrema)
        self.time_min_jerk = min_peak_time_jerk
    
    # calculate signal to noise ratio for jerk
    def calc_jerk_signal_noise(self):
        
        # calculate the displacement by integrating the velocity
        tbase_start = -360
        tbase_end = 0
        base_start_idx = np.where(self.time_data[0] >= tbase_start)[0][0] # find index of start of baseline
        base_end_idx = np.where(self.time_data[0] <= tbase_end)[0][-1] # find index of end of baseline
        base_jerk = np.array(self.jerk_data[0][base_start_idx:base_end_idx])
        
        # find extrema of jerk
        data_start = 20
        data_end = 70
        data_start_idx = np.where(self.time_data[0] >= data_start)[0][0]
        data_end_idx = np.where(self.time_data[0] <= data_end)[0][-1]
        data_jerk = np.array(self.jerk_data[0][data_start_idx:data_end_idx])
        data_max_jerk = np.max(data_jerk)

        # Ensure there are always 3000 data points
        desired_points = 9000
        actual_points = base_end_idx - base_start_idx
        extra_points = desired_points - actual_points

        if extra_points > 0:
            # Need to extend the data range
            base_end_idx += extra_points
        elif extra_points < 0:
            # Need to reduce the data range
            print('Error: invalid temporal range given')

        self.baseline_jerk = np.array(self.jerk_data[0][base_start_idx:base_end_idx])
        self.baseline_jerk_rms = np.sqrt(np.mean(base_jerk ** 2))  # recalculate RMS
        self.jerk_SNR = data_max_jerk / self.baseline_jerk_rms # calculate signal to noise ratio       
        
    def calc_time_vel_0_2_4(self):
        
        # time of velocity threshold 
        # v = 0 m/s, 2 m/s, and v = 4 m/s
        vel_0_idx = np.where(self.velocity_data[0] <= 0)[0][-1]
        vel_2_idx = np.where(self.velocity_data[0] <= 2)[0][-1]
        vel_4_idx = np.where(self.velocity_data[0] <= 4)[0][-1]
        self.time_vel_0 = self.time_data[0][vel_0_idx]
        self.time_vel_2 = self.time_data[0][vel_2_idx]
        self.time_vel_4 = self.time_data[0][vel_4_idx]
#---

#---
### data imports and analysis setup
#
# shot dependent time shifts
tshift94 = 0
tshift95 = 0
tshift96 = 0
tshift97 = 0

### import data
# file handling 
# import data from SMASH output files

data_location = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/'

## ED1 shot 10194
file94 = data_location + 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data94 = np.loadtxt(file94, skiprows=30, usecols=(0,1))
t94 = (data94[:,0]*1e9)[0:-1-400]+tshift94
vel94 = data94[:,1][0:-1-400]

# ED1 shot 10195
file95 = data_location + 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data95 = np.loadtxt(file95, skiprows=30, usecols=(0,1))
t95 = (data95[:,0]*1e9)[0:-1-400]+tshift95
vel95 = data95[:,1][0:-1-400]

# ED1 shot 10196
file96 = data_location + 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data96 = np.loadtxt(file96, skiprows=30, usecols=(0,1))
t96 = (data96[:,0]*1e9)[0:-1-400]+tshift96
vel96 = data96[:,1][0:-1-400]

# ED1 shot 10197
file97 = data_location + 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data97 = np.loadtxt(file97, skiprows=30, usecols=(0,1))
t97 = (data97[:,0]*1e9)[0:-1-400]+tshift97
vel97 = data97[:,1][0:-1-400]
#---

#---
#
# function calls
#
### analysis for 10194
PDV94 = PDVshot('10194', 4.8)
PDV94.add_vel_hist(t94, vel94)
PDV94.smooth_data(13, 1)
PDV94.calc_SG_derivatives(80)
PDV94.calc_baseline_noise()
PDV94.calc_compression_disp()
PDV94.calc_compression_t_test()
PDV94.calc_extrema()
PDV94.calc_peaks_derivatives()
PDV94.calc_jerk_signal_noise()
PDV94.calc_time_vel_0_2_4()

### analysis for 10195
PDV95 = PDVshot('10195', 4.8)
PDV95.add_vel_hist(t95, vel95)
PDV95.smooth_data(13, 1)
PDV95.calc_SG_derivatives(80)
PDV95.calc_baseline_noise()
PDV95.calc_compression_disp()
PDV95.calc_compression_t_test()
PDV95.calc_extrema()
PDV95.calc_peaks_derivatives()
PDV95.calc_jerk_signal_noise()
PDV95.calc_time_vel_0_2_4()

### analysis for 10196
PDV96 = PDVshot('10196', 4.8)
PDV96.add_vel_hist(t96, vel96)
PDV96.smooth_data(13, 1)
PDV96.calc_SG_derivatives(80)
PDV96.calc_baseline_noise()
PDV96.calc_compression_disp()
PDV96.calc_compression_t_test()
PDV96.calc_extrema()
PDV96.calc_peaks_derivatives()
PDV96.calc_jerk_signal_noise()
PDV96.calc_time_vel_0_2_4()

### analysis for 10197
PDV97 = PDVshot('10197', 4.8)
PDV97.add_vel_hist(t97, vel97)
PDV97.smooth_data(13, 1)
PDV97.calc_SG_derivatives(80)
PDV97.calc_baseline_noise()
PDV97.calc_compression_disp()
PDV97.calc_compression_t_test()
PDV97.calc_extrema()
PDV97.calc_peaks_derivatives()
PDV97.calc_jerk_signal_noise()
PDV97.calc_time_vel_0_2_4()
#---

#---
### results 
#
# examples of accessing and printing out values of interest for each shot
#

# baseline velocity rms
print('Baseline (-360 to 0 ns) velocity RMS:')
print('Shot 10194: ',round(PDV94.baseline_vel_rms,2), 'm/s')
print('Shot 10195: ',round(PDV95.baseline_vel_rms,2), 'm/s')
print('Shot 10196: ',round(PDV96.baseline_vel_rms,2), 'm/s')
print('Shot 10197: ',round(PDV97.baseline_vel_rms,2), 'm/s')

# baseline jerk rms
print('Baseline (-360 to 0 ns) jerk RMS:')
print('Shot 10194: ',round(PDV94.baseline_jerk_rms,2), 'nm/ns^3')
print('Shot 10195: ',round(PDV95.baseline_jerk_rms,2), 'nm/ns^3')
print('Shot 10196: ',round(PDV96.baseline_jerk_rms,2), 'nm/ns^3')
print('Shot 10197: ',round(PDV97.baseline_jerk_rms,2), 'nm/ns^3')

# time of max negative velocity
print('Time of max negative velocity:')
print('Shot 10194: ',round(PDV94.time_max_neg_vel,2), 'ns')
print('Shot 10195: ',round(PDV95.time_max_neg_vel,2), 'ns')
print('Shot 10196: ',round(PDV96.time_max_neg_vel,2), 'ns')
print('Shot 10197: ',round(PDV97.time_max_neg_vel,2), 'ns')

# max negative velocity
print('Max negative velocity:')
print('Shot 10194: ',round(PDV94.max_neg_vel,2), 'm/s')
print('Shot 10195: ',round(PDV95.max_neg_vel,2), 'm/s')
print('Shot 10196: ',round(PDV96.max_neg_vel,2), 'm/s')
print('Shot 10197: ',round(PDV97.max_neg_vel,2), 'm/s')

# compression displacement
print('Compression displacement:')
print('Shot 10194: ',round(PDV94.comp_disp,2), 'nm')
print('Shot 10195: ',round(PDV95.comp_disp,2), 'nm')
print('Shot 10196: ',round(PDV96.comp_disp,2), 'nm')
print('Shot 10197: ',round(PDV97.comp_disp,2), 'nm')

# time of velocity = 4 m/s
print('Time of velocity = 4 m/s:')
print('Shot 10194: ',round(PDV94.time_vel_4,2), 'ns')
print('Shot 10195: ',round(PDV95.time_vel_4,2), 'ns')
print('Shot 10196: ',round(PDV96.time_vel_4,2), 'ns')
print('Shot 10197: ',round(PDV97.time_vel_4,2), 'ns')

# time of local max jerk
print('Time of local max jerk:')
print('Shot 10194: ',round(PDV94.time_max_jerk,2), 'ns')
print('Shot 10195: ',round(PDV95.time_max_jerk,2), 'ns')
print('Shot 10196: ',round(PDV96.time_max_jerk,2), 'ns')
print('Shot 10197: ',round(PDV97.time_max_jerk,2), 'ns')

# time of local max acceleration
print('Time of local max acceleration:')
print('Shot 10194: ',round(PDV94.time_max_accel,2), 'ns')
print('Shot 10195: ',round(PDV95.time_max_accel,2), 'ns')
print('Shot 10196: ',round(PDV96.time_max_accel,2), 'ns')
print('Shot 10197: ',round(PDV97.time_max_accel,2), 'ns')

# time of local min jerk
print('Time of local min jerk:')
print('Shot 10194: ',round(PDV94.time_min_jerk,2), 'ns')
print('Shot 10195: ',round(PDV95.time_min_jerk,2), 'ns')
print('Shot 10196: ',round(PDV96.time_min_jerk,2), 'ns')
print('Shot 10197: ',round(PDV97.time_min_jerk,2), 'ns')

# duration of melt
print('Duration of melt (time between extrema of jerk):')
print('Shot 10194: ',round(PDV94.duration_melt,1), 'ns')
print('Shot 10195: ',round(PDV95.duration_melt,1), 'ns')
print('Shot 10196: ',round(PDV96.duration_melt,1), 'ns')
print('Shot 10197: ',round(PDV97.duration_melt,1), 'ns')


# average and standard deviation duration of melt
avg_duration_melt = (PDV94.duration_melt + PDV95.duration_melt + PDV96.duration_melt + PDV97.duration_melt)/4
stdev_duration_melt = np.std([PDV94.duration_melt, PDV95.duration_melt, PDV96.duration_melt, PDV97.duration_melt])
print(r'Average duration of melt over all four shots: %.1f ns with STDEV of %.1f ns' %(avg_duration_melt, stdev_duration_melt))









