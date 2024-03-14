
"""
@author: Aidanklemmer
University of Nevada, Reno
Aidanklemmer@outlook.com
3/9/24
"""

### PDV post-processing 
# derivative analysis
# and derivative triplet plotting

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import scipy.stats as stats
from scipy.signal import savgol_filter


# select shot
shot = ['97']

# create figure
fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(6,7), sharex=True)

# global line and marker options
markersize = 0.6
linewidth = 1

# shot dependent time shifts
tshift94 = 0
tshift95 = 0
tshift96 = 0
tshift97 = 0

# digitizer speed and STFT window spacing
alpha = 40e-12 # 1/25 GHz = 0.04 ns

# functions
# ------------
# function for velocity offset
def calc_offset(time, vel, toff_start, toff_end):
    # Find indices corresponding to region of interest
    i_start = np.searchsorted(time, toff_start)
    i_end = np.searchsorted(time, toff_end)
    # Calculate average over region
    avg = np.mean(vel[i_start:i_end])
    return avg

# function for moving average 
def smooth_conv(time, vel, window_size):
    times = time
    smoothed_vel = np.convolve(vel, np.ones(window_size)/window_size, mode='same')
    return times,smoothed_vel

# function for Savitzky-Golay (SG) derivatives and filters
def calc_SG_deriv(time, vel, win, order):
    SG_filter = sc.signal.savgol_filter(vel, win, order, deriv=1, delta=0.04e-9)
    SG_filter_2 = sc.signal.savgol_filter(vel, win, order, deriv=2, delta=0.04e-9)
    deriv = SG_filter * 1e-9
    deriv_2 = SG_filter_2 * 1e-18
    return time, deriv, deriv_2

# function for calculating physical parameters from vel, accel, and jerk
def calc_parameters(t_avg_deriv, time, vel, deriv, deriv_2):
    
    # linear interpolation
    f1 = interp1d(t_avg_deriv, deriv)
    f2 = interp1d(t_avg_deriv, deriv_2)

    # interpolate at some new points
    # find local max and min
    t_start = 45
    t_end = 72
    
    # acceleration - 1st derivative
    x1 = np.arange(t_start, t_end, 1e-4)
    y1_new = f1(x1)
    # jerk - 2nd derivative
    x2 = np.arange(t_start, t_end, 1e-4)
    y2_new = f2(x2)

    # find zero crossings
    zero_crossings_deriv = np.where(np.diff(np.sign(y1_new)))[0] # acceleration
    zero_crossings_deriv2 = np.where(np.diff(np.sign(y2_new)))[0] # jerk
    
    # find extrema using scipy.signal.find_peaks
    # acceleration
    peaks_deriv, _ = find_peaks(y1_new)
    valleys_deriv, _ = find_peaks(-np.array(y1_new))
    extrema_deriv = np.sort(np.concatenate((peaks_deriv, valleys_deriv))) 
    
    # find extrema using scipy.signal.find_peaks
    # jerk
    peaks_deriv2, _ = find_peaks(y2_new)
    valleys_deriv2, _ = find_peaks(-np.array(y2_new))
    extrema_deriv2 = np.sort(np.concatenate((peaks_deriv2, valleys_deriv2)))
    
    # find the indices that correspond to the time range
    # time domain (x1, and x2 data) are the same for both first and second derivative
    idx_start = np.where(x2 >= t_start)[0][0]
    idx_end = np.where(x2 <= t_end)[0][-1]
    
    # calculate zero crossing and extrema of acceleration
    # take only data that falls within the time range
    zc_range_accel = zero_crossings_deriv[(zero_crossings_deriv >= idx_start) & (zero_crossings_deriv <= idx_end)] # not currently used
    extrema_range_accel = extrema_deriv[(extrema_deriv >= idx_start) & (extrema_deriv <= idx_end)]
    
    # calculate zero crossing and extrema of jerk
    # take only data that falls within the time range
    zc_range_jerk = zero_crossings_deriv2[(zero_crossings_deriv2 >= idx_start) & (zero_crossings_deriv2 <= idx_end)]
    extrema_range_jerk = extrema_deriv2[(extrema_deriv2 >= idx_start) & (extrema_deriv2 <= idx_end)]
    
    # find max values of acceleration
    max_peak_time_accel = x1[extrema_range_accel[np.argmax(y1_new[extrema_range_accel])]]

    # find max and min values of jerk
    max_peak_time_jerk = x2[extrema_range_jerk[np.argmax(y2_new[extrema_range_jerk])]]
    min_peak_time_jerk = x2[extrema_range_jerk[np.argmin(y2_new[extrema_range_jerk])]]
    
    # find duration of time between extrema of jerk
    duration = min_peak_time_jerk - max_peak_time_jerk
    
    # local minimum of acceleration after peak acceleration
    min_accel_after_max_idx = np.where(np.array(x1)[extrema_range_accel] > max_peak_time_accel)

    
    # calculate time of max compression
    # first zero crossing of jerk before local max of jerk
    max_comp_loc = np.where(np.array(x2)[zc_range_jerk] <= max_peak_time_jerk)
    t_max_comp = np.array(x2)[zc_range_jerk][max_comp_loc][-1]
    
    # calculate max negative velocity 
    tc_start = 45
    tc_end = 75
    
    # find compression region population
    idc_start = np.where(time >= tc_start)[0][0] # find the indices that correspond to the time range
    idc_end = np.where(time <= tc_end)[0][-1]
    comp_max_vel = np.min(vel[idc_start:idc_end]) # find max neg vel, this is the min vel value in the range
    ind_comp_max = np.where(vel <= np.min(vel[idc_start:idc_end]))
    center_index = ind_comp_max[0][0]  # Index of the center value
    stat_shift = -75  # 3 ns shift earlier in time. This is arbitrary choice
    offsets = [-81+stat_shift,  0+stat_shift, 81+stat_shift] # define the offsets/spacing for the 3 pt comp region population
    vel_3pt_comp_pop = vel[[center_index + offset for offset in offsets]] # calculate the 3 pt comp population
    vel_3pt_comp_pop = [round(val, 3) for val in vel_3pt_comp_pop] # round
    time_center_3pt_comp_pop = time[center_index] # calculate the center time of the 3 pt comp population
    
    # compression displacement
    tcomp_start = 20
    comp_end = np.where(vel  <= 0)[0][-1]
    comp_start = np.where(time >= tcomp_start)[0][0]
    comp_reg = vel[comp_start:comp_end]
    comp_disp = sc.integrate.simpson(comp_reg, dx=alpha)
    
    
    # print out values
    # acceleration
    print('Time of max accel: ', str(round(float(max_peak_time_accel),3)), 'ns')
    print('Time of min accel: ', str(round(float(np.array(x1)[extrema_range_accel][min_accel_after_max_idx][0]),3)), 'ns')
    print('Max accel: ', str(round(float(y1_new[extrema_range_accel[np.argmax(y1_new[extrema_range_accel])]]),3)), 'nm/ns^2')
    print('Min accel: ', str(round(float(np.array(y1_new)[extrema_range_accel][min_accel_after_max_idx][0]),3)), 'nm/ns^2')
    
    # jerk
    print('Time of max jerk: ', str(round(float(max_peak_time_jerk),3)), 'ns')
    print('Time of min jerk: ', str(round(float(min_peak_time_jerk),3)), 'ns')
    print('Max jerk: ', str(round(float(y2_new[extrema_range_jerk[np.argmax(y2_new[extrema_range_jerk])]]),3)), 'nm/ns^3')
    print('Min jerk: ', str(round(float(y2_new[extrema_range_jerk[np.argmin(y2_new[extrema_range_jerk])]]),3)), 'nm/ns^3')
    
    print('Compression region population: ', vel_3pt_comp_pop, 'm/s')
    print('Time of max negative velocity (center of comp. pop.): ', str(round(time_center_3pt_comp_pop,3)), 'ns')
    print('Time of comp end (max comp displacement): ', str(round(time[comp_end],3)), 'ns')
    
    
    ## plotting 
    #---
    # max acceleration
    ax2.plot(max_peak_time_accel, y1_new[extrema_range_accel[np.argmax(y1_new[extrema_range_accel])]], 'gs', label='Local max acceleration', ms=3)
    
    # local min acceleration after local max acceleration
    ax2.plot(np.array(x1)[extrema_range_accel][min_accel_after_max_idx][0], np.array(y1_new)[extrema_range_accel][min_accel_after_max_idx][0], 'g*',label='Local min acceleration',ms=4)
    
    # zero crossings of jerk
    #ax3.plot(np.array(x2)[zc_range_jerk], np.array(y2_new)[zc_range_jerk], 'co', ms=3, label='Second Derivative - Zero crossings')
    # extema of jerk
    #ax3.plot(np.array(x2)[extrema_range_jerk], np.array(y2_new)[extrema_range_jerk], 'rs', ms=3,label='Second Derivative - Extrema')
    
    # max jerk
    ax3.plot(max_peak_time_jerk, y2_new[extrema_range_jerk[np.argmax(y2_new[extrema_range_jerk])]], 'gs',label='Local max jerk',ms=3)
    # min jerk
    ax3.plot(min_peak_time_jerk, y2_new[extrema_range_jerk[np.argmin(y2_new[extrema_range_jerk])]], 'g*',label='Local min jerk',ms=4)

    # first zero crossing before melt - max compression displacement
    ax1.plot(time[comp_end], vel[comp_end], 'go',label='Max displacement (v = 0)',ms=3)
    
    # max negative velocity 
    ax1.plot(time[center_index], vel[center_index], 'g^', label='Max negative velocity', ms=3)
    #---

    return duration, t_max_comp, comp_max_vel, comp_disp, comp_reg, time_center_3pt_comp_pop, vel_3pt_comp_pop



#---

## calculate noise of baseline (flucation of velocity in baseline region)
def calc_baseline_noise(time, vel):
    # calculate the displacement by integrating the velocity
    tbase_start = -100
    tbase_end = 20
    base_start = np.where(time >= tbase_start)[0][0]
    base_end = np.where(time <= tbase_end)[0][-1]
    base_vel = vel[base_start:base_end]
    base_vel_avg = np.mean(base_vel)
    base_vel_stdev = np.std(base_vel)
    
    return base_vel_avg, base_vel_stdev, base_vel


# calculate signal to noise ratio for jerk
def calc_jerk_signal_noise(time, vel, deriv_2):
    # calculate the displacement by integrating the velocity
    tbase_start = -100
    tbase_end = 20
    base_start_idx = np.where(time >= tbase_start)[0][0] # find index of start of baseline
    base_end_idx = np.where(time <= tbase_end)[0][-1] # find index of end of baseline
    base_jerk = np.array(deriv_2[base_start_idx:base_end_idx])
    base_jerk_rms = np.sqrt(np.mean(base_jerk ** 2))  # calculate RMS
    
    # find extrema of jerk
    data_start = 20
    data_end = 70
    data_start_idx = np.where(time >= data_start)[0][0]
    data_end_idx = np.where(time <= data_end)[0][-1]
    data_jerk = np.array(deriv_2[data_start_idx:data_end_idx])
    data_max_jerk = np.max(data_jerk)

    # Ensure there are always 3000 data points
    desired_points = 3000
    actual_points = base_end_idx - base_start_idx
    extra_points = desired_points - actual_points

    if extra_points > 0:
        # Need to extend the data range
        base_end_idx += extra_points
    elif extra_points < 0:
        # Need to reduce the data range
        print('Error: invalid temporal range given')

    base_jerk = np.array(deriv_2[base_start_idx:base_end_idx])
    base_jerk_rms = np.sqrt(np.mean(base_jerk ** 2))  # recalculate RMS
    jerk_sig_noise_ratio = data_max_jerk / base_jerk_rms # calculate signal to noise ratio

    return base_jerk_rms, data_max_jerk, jerk_sig_noise_ratio


# calculate t-test statistics 
# Welch t-test
def calc_stats(base, comp_max3):
    # Perform T-test
    t_statistic3, p_value3 = stats.ttest_ind(comp_max3, np.zeros(3), equal_var=False)  
    
    return t_statistic3, p_value3


# plot the SMASH SIRHEN ROI along with the velocity-history
def plot_ROI(shot_num,center_times,smoothed_velocity):
    
    # create figure
    fig2, (ax_roi) = plt.subplots(1,1,figsize=(5,4))

    # width "height" of ROI in m/s
    roi_width = 160
    
    
    if '10197' in shot_num:
        # Define the ROI data
        ROI_data = [
            [-1e-7, 0 , roi_width], 
            [0, 0, roi_width], 
            [54e-9, -5, roi_width], 
            [57.2e-9, -0.5, roi_width], 
            [59.6e-9, 5.2, roi_width], 
            [61.2e-9, 19.8, roi_width], 
            [63.2e-9, 59.5, roi_width], 
            [65.2e-9, 97.1, roi_width], 
            [67.2e-9, 128, roi_width], 
            [68e-9, 140.2, roi_width], 
            [69.2e-9, 159, roi_width], 
            [71.2e-9, 197.7, roi_width], 
            [73.2e-9, 241.2, roi_width], 
            [76.4e-9, 315.6, roi_width], 
            [78.4e-9, 373.6, roi_width], 
            [80.4e-9, 444, roi_width], 
            [82.2e-9, 529.4, roi_width], 
            [84.4e-9, 629.8, roi_width], 
            [87.2e-9, 775.2, roi_width], 
            [90.4e-9, 1022, roi_width], 
            [94.4e-9, 1476, roi_width], 
            [110e-9, 1500, roi_width]]
    
    if '10196' in shot_num:
        ROI_data = [
            [-1e-7, 0, roi_width],
            [0, 0, roi_width],
            [54e-9, -5, roi_width],
            [57.2e-9, -3, roi_width],
            [61.2e-9, 14, roi_width],
            [63.2e-9, 49.6, roi_width],
            [65.2e-9, 82.3, roi_width],
            [67.2e-9, 113.6, roi_width],
            [68e-9, 126.3, roi_width],
            [69.2e-9, 144.2, roi_width],
            [71.2e-9, 175.2, roi_width],
            [73.2e-9, 211, roi_width],
            [76.4e-9, 276, roi_width],
            [78.4e-9, 328, roi_width],
            [80.4e-9, 389, roi_width],
            [84.4e-9, 560, roi_width],
            [87.2e-9, 715, roi_width],
            [90.4e-9, 899, roi_width],
            [94.4e-9, 1326, roi_width],
            [110e-9, 1400, roi_width]]
     
    if '10195' in shot_num: 
        ROI_data = [
            [-1e-7, 0, roi_width],
            [0, 0, roi_width],
            [54e-9, -3.5, roi_width],
            [57e-9, -2.85, roi_width],
            [61.2e-9, 15.5, roi_width],
            [63.2e-9, 40.9, roi_width],
            [65.2e-9, 88.5, roi_width],
            [67.2e-9, 128.4, roi_width],
            [68e-9, 136.8, roi_width],
            [69.2e-9, 145, roi_width],
            [71.2e-9, 165, roi_width],
            [73.2e-9, 189, roi_width],
            [74.4e-9, 212, roi_width],
            [76e-9, 248, roi_width],
            [78.2e-9, 400, roi_width],
            [80e-9, 420, roi_width],
            [84.4e-9, 662, roi_width],
            [90.4e-9, 1084, roi_width],
            [94.4e-9, 1267, roi_width],
            [110e-9, 1400, roi_width]]
    
    if '10194' in shot_num:
        ROI_data = [
            [-1e-7, 0, roi_width],
            [0, 0, roi_width],
            [54e-9, -5, roi_width],
            [57e-9, 0, roi_width],
            [61.2e-9, 10, roi_width],
            [63.2e-9, 48, roi_width],
            [65.2e-9, 93.5, roi_width],
            [67.2e-9, 115, roi_width],
            [68e-9, 124, roi_width],
            [69.2e-9, 134.9, roi_width],
            [71.2e-9, 171, roi_width],
            [73.2e-9, 213, roi_width],
            [76.2e-9, 295, roi_width],
            [78.2e-9, 350, roi_width],
            [80.2e-9, 419.3, roi_width],
            [84.2e-9, 665, roi_width],
            [90.2e-9, 1081, roi_width],
            [94.2e-9, 1328, roi_width],
            [110e-9, 1400, roi_width]]
    
    # Convert the data to numpy array for easier manipulation
    ROI_data = np.array(ROI_data)
    
    # Extract time, velocity, and width columns
    time = ROI_data[:, 0]*1e9
    velocity = ROI_data[:, 1]
    roi_widths = ROI_data[:, 2]
    
    # Calculate the minimum and maximum values of the ROI
    roi_min = velocity - roi_widths / 2
    roi_max = velocity + roi_widths / 2
    
    # Plot the results
    ax_roi.plot(time, roi_min, 'k--', label='ROI Min')
    ax_roi.plot(time, roi_max, 'k--', label='ROI Max')
    ax_roi.plot(center_times,smoothed_velocity,'-',markersize=markersize,linewidth=linewidth, color = [1,0.7,0.3])
    
    ax_roi.set_xlim(0,80)
    ax_roi.set_ylim(-150,500)
    
    ax_roi.set_xlabel('Time [ns]', fontsize=8)
    ax_roi.set_ylabel('Velocity [m/s]', fontsize=8)
    ax_roi.set_title('Shot %.f, 4.8 ns tau, ROI with %s m/s width' %(int(shot_num), roi_width), fontsize=8)
    fig2.set_dpi(300)
        


# ------------

### ED1 10194
if '94' in shot:

        # file handling 
        # import data from SMASH output files
        file94='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_new.csv'
        data94 = np.loadtxt(file94, skiprows=30, usecols=(0,1))
        t94 = (data94[:,0]*1e9)[0:-1-400]+tshift94
        vel94 = data94[:,1][0:-1-400]

        #---
        # smooth data
        # Calculate moving average using convolution
        mov_avg_win = 13
        center_times94_1, smoothed_velocity94_1 = smooth_conv(t94, vel94, mov_avg_win)
        center_times94, smoothed_velocity94 = smooth_conv(center_times94_1, smoothed_velocity94_1, mov_avg_win)
        
        # SG filter
        winSG94 = 80
        orderSG = 3
        SG_smooth_data94 = savgol_filter(smoothed_velocity94, winSG94, orderSG)
        
        #---
        ## calculate offset
        toff_start = -100
        toff_end = 20
        avg94 = calc_offset(center_times94, SG_smooth_data94, toff_start, toff_end)
        smoothed_velocity94_SG = SG_smooth_data94 - avg94
        
        #---
        ## plotting

        lab_94 = r'ED1 PDV Velocity #10194 - 4.8 ns tau with Boxcar window, Gaussian peakfinding, smoothed: double %.0f pt moving avg, %.0f pt SG filter' %(mov_avg_win, winSG94) # label
        # raw
        ax1.plot(t94,vel94,'-',markersize=markersize,linewidth=linewidth, label='ED1 PDV Velocity #10194 - 4.8 ns tau with Boxcar window, Gaussian peakfinding', color=(0, 0, 0))
        
        # filtered
        ax1.plot(center_times94,smoothed_velocity94_SG,'-',markersize=markersize,linewidth=linewidth, label=lab_94.format(str(tshift94)), color=(1, 0, 0))
        #---
        
        #---
        ## calculate numerical instantaneous derivative via SG
        t_avg_deriv, deriv, deriv_2 = calc_SG_deriv(center_times94, smoothed_velocity94, winSG94, orderSG)
        
        ax2.plot(t_avg_deriv, deriv, color= 'black', label='Savitzky-Golay derivative, cubic fit with %.f pt SG window' %(winSG94))
        ax3.plot(t_avg_deriv, deriv_2,color= 'black',label='Savitzky-Golay 2nd derivative, cubic fit with %.f pt SG window' %(winSG94))
        
        # call functions
        duration, t_max_comp, comp_max_vel, comp_disp, com_reg, time_center_3pt_comp_pop, vel_3pt_comp_pop = calc_parameters(t_avg_deriv, center_times94, smoothed_velocity94_SG, deriv, deriv_2)
        baseline_avg, baseline_stdev, baseline = calc_baseline_noise(center_times94, smoothed_velocity94_SG)
        t_statistic3, p_value3 = calc_stats(baseline, vel_3pt_comp_pop)
        baseline_rms_jerk, data_max_jerk, jerk_sig_noise_ratio  = calc_jerk_signal_noise(center_times94, smoothed_velocity94_SG, deriv_2)
    
        
        print('Offset (-100 to 20 ns): ', str(round(avg94,3)), 'm/s')
        print('Time between extrema of jerk (max-min):',str(round(duration,3)), 'ns')
        print('Max negative velocity: ', str(round(comp_max_vel,3)), 'm/s')
        print('Compression displacement:', str(round(-comp_disp*1e9,3)), 'nm')
        print('Baseline vel noise (-100 to 20 ns) and STDEV:', str(round(baseline_avg,5)), str(round(baseline_stdev,3)), 'm/s')
        print('T stat and P value for compression region (3 pt around max neg. vel.):', str(round(t_statistic3,3)), str(round(p_value3,3)))
        print('Baseline jerk (-100 to 20 ns) RMS: ', str(round(baseline_rms_jerk,3)))
        print('Jerk signal to noise ratio: ', str(round(jerk_sig_noise_ratio,3)))
        
        ## option to generate a second figure and plot the ROI with the velocity-history
        #plot_ROI('10194',center_times94,smoothed_velocity94_SG)



### ED1 10195
if '95' in shot:
        
        # file handling 
        # import data from SMASH output files
        file95='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_new.csv'
        data95 = np.loadtxt(file95, skiprows=30, usecols=(0,1))
        t95 = (data95[:,0]*1e9)[0:-1-400]+tshift95
        vel95 = data95[:,1][0:-1-400]

        #---
        # smooth data
        # Calculate moving average using convolution
        mov_avg_win = 11
        center_times95_1, smoothed_velocity95_1 = smooth_conv(t95, vel95, mov_avg_win)
        center_times95, smoothed_velocity95 = smooth_conv(center_times95_1, smoothed_velocity95_1, mov_avg_win)
        
        # SG filter
        winSG95 = 80
        orderSG = 3
        SG_smooth_data95 = savgol_filter(smoothed_velocity95, winSG95, orderSG)
        
        #---
        ## calculate offset
        toff_start = -100
        toff_end = 20
        avg95 = calc_offset(center_times95, SG_smooth_data95, toff_start, toff_end)
        smoothed_velocity95_SG = SG_smooth_data95 - avg95
        
        #---
        ## plotting

        lab_95 = r'ED1 PDV Velocity #10195 - 4.8 ns tau with Boxcar window, Gaussian peakfinding, smoothed: double %.0f pt moving avg, %.0f pt SG filter' %(mov_avg_win, winSG95) # label
        # raw
        ax1.plot(t95,vel95,'-',markersize=markersize,linewidth=linewidth, label='ED1 PDV Velocity #10195 - 4.8 ns tau with Boxcar window, Gaussian peakfinding', color=(0, 0, 0))
        
        # filtered
        ax1.plot(center_times95,smoothed_velocity95_SG,'-',markersize=markersize,linewidth=linewidth, label=lab_95.format(str(tshift95)), color=(1, 0, 1))
        #---
        
        #---
        ## calculate numerical instantaneous derivative via SG
        t_avg_deriv, deriv, deriv_2 = calc_SG_deriv(center_times95, smoothed_velocity95, winSG95, orderSG)
        
        ax2.plot(t_avg_deriv, deriv, color= 'black', label='Savitzky-Golay derivative, cubic fit with %.f pt SG window' %(winSG95))
        ax3.plot(t_avg_deriv, deriv_2,color= 'black',label='Savitzky-Golay 2nd derivative, cubic fit with %.f pt SG window' %(winSG95))
        
        # call functions
        duration, t_max_comp, comp_max_vel, comp_disp, com_reg, time_center_3pt_comp_pop, vel_3pt_comp_pop = calc_parameters(t_avg_deriv, center_times95, smoothed_velocity95_SG, deriv, deriv_2)
        baseline_avg, baseline_stdev, baseline = calc_baseline_noise(center_times95, smoothed_velocity95_SG)
        t_statistic3, p_value3 = calc_stats(baseline, vel_3pt_comp_pop)
        baseline_rms_jerk, data_max_jerk, jerk_sig_noise_ratio  = calc_jerk_signal_noise(center_times95, smoothed_velocity95_SG, deriv_2)
        
        
        print('Offset (-100 to 20 ns): ', str(round(avg95,3)), 'm/s')
        print('Time between extrema of jerk (max-min):',str(round(duration,3)), 'ns')
        print('Max negative velocity: ', str(round(comp_max_vel,3)), 'm/s')
        print('Compression displacement:', str(round(-comp_disp*1e9,3)), 'nm')
        print('Baseline vel noise (-100 to 20 ns) and STDEV:', str(round(baseline_avg,5)), str(round(baseline_stdev,3)), 'm/s')
        print('T stat and P value for compression region (3 pt around max neg. vel.):', str(round(t_statistic3,3)), str(round(p_value3,3)))
        print('Baseline jerk (-100 to 20 ns) RMS: ', str(round(baseline_rms_jerk,3)))
        print('Jerk signal to noise ratio: ', str(round(jerk_sig_noise_ratio,3)))
        
        ## option to generate a second figure and plot the ROI with the velocity-history
        #plot_ROI('10195',center_times95,smoothed_velocity95_SG)
        
        
        
        
### ED1 10196   
if '96' in shot:
        
        # file handling 
        # import data from SMASH output files
        file96='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_new.csv'
        data96 = np.loadtxt(file96, skiprows=30, usecols=(0,1))
        t96 = (data96[:,0]*1e9)[0:-1-400]+tshift96
        vel96 = data96[:,1][0:-1-400]

        #---
        # smooth data
        # Calculate moving average using convolution
        mov_avg_win = 11
        center_times96_1, smoothed_velocity96_1 = smooth_conv(t96, vel96, mov_avg_win)
        center_times96, smoothed_velocity96 = smooth_conv(center_times96_1, smoothed_velocity96_1, mov_avg_win)
        
        # SG filter
        winSG96 = 80
        orderSG = 3
        SG_smooth_data96 = savgol_filter(smoothed_velocity96, winSG96, orderSG)
        
        #---
        ## calculate offset
        toff_start = -100
        toff_end = 20
        avg96 = calc_offset(center_times96, SG_smooth_data96, toff_start, toff_end)
        smoothed_velocity96_SG = SG_smooth_data96 - avg96
        
        #---
        ## plotting

        lab_96 = r'ED1 PDV Velocity #10196 - 4.8 ns tau with Boxcar window, Gaussian peakfinding, smoothed: double %.0f pt moving avg, %.0f pt SG filter' %(mov_avg_win, winSG96) # label
        # raw
        ax1.plot(t96,vel96,'-',markersize=markersize,linewidth=linewidth, label='ED1 PDV Velocity #10196 - 4.8 ns tau with Boxcar window, Gaussian peakfinding', color=(0, 0, 0))
        
        # filtered
        ax1.plot(center_times96,smoothed_velocity96_SG,'-',markersize=markersize,linewidth=linewidth, label=lab_96.format(str(tshift96)), color=(0, 0, 1))
        #---
        
        #---
        ## calculate numerical instantaneous derivative via SG
        t_avg_deriv, deriv, deriv_2 = calc_SG_deriv(center_times96, smoothed_velocity96, winSG96, orderSG)
        
        ax2.plot(t_avg_deriv, deriv, color= 'black', label='Savitzky-Golay derivative, cubic fit with %.f pt SG window' %(winSG96))
        ax3.plot(t_avg_deriv, deriv_2,color= 'black',label='Savitzky-Golay 2nd derivative, cubic fit with %.f pt SG window' %(winSG96))
        
        # call functions
        duration, t_max_comp, comp_max_vel, comp_disp, com_reg, time_center_3pt_comp_pop, vel_3pt_comp_pop = calc_parameters(t_avg_deriv, center_times96, smoothed_velocity96_SG, deriv, deriv_2)
        baseline_avg, baseline_stdev, baseline = calc_baseline_noise(center_times96, smoothed_velocity96_SG)
        t_statistic3, p_value3 = calc_stats(baseline, vel_3pt_comp_pop)
        baseline_rms_jerk, data_max_jerk, jerk_sig_noise_ratio  = calc_jerk_signal_noise(center_times96, smoothed_velocity96_SG, deriv_2)
        
        
        print('Offset (-100 to 20 ns): ', str(round(avg96,3)), 'm/s')
        print('Time between extrema of jerk (max-min):',str(round(duration,3)), 'ns')
        print('Max negative velocity: ', str(round(comp_max_vel,3)), 'm/s')
        print('Compression displacement:', str(round(-comp_disp*1e9,3)), 'nm')
        print('Baseline vel noise (-100 to 20 ns) and STDEV:', str(round(baseline_avg,5)), str(round(baseline_stdev,3)), 'm/s')
        print('T stat and P value for compression region (3 pt around max neg. vel.):', str(round(t_statistic3,3)), str(round(p_value3,3)))
        print('Baseline jerk (-100 to 20 ns) RMS: ', str(round(baseline_rms_jerk,3)))
        print('Jerk signal to noise ratio: ', str(round(jerk_sig_noise_ratio,3)))
        
        ## option to generate a second figure and plot the ROI with the velocity-history
        #plot_ROI('10196',center_times96,smoothed_velocity96_SG)

    

### ED1 10197    
if '97' in shot:   
        
        # file handling 
        # import data from SMASH output files
        file97='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_new.csv'
        data97 = np.loadtxt(file97, skiprows=30, usecols=(0,1))
        t97 = (data97[:,0]*1e9)[0:-1-400]+tshift97
        vel97 = data97[:,1][0:-1-400]

        #---
        # smooth data
        # Calculate moving average using convolution
        mov_avg_win = 11
        center_times97_1, smoothed_velocity97_1 = smooth_conv(t97, vel97, 11)
        center_times97, smoothed_velocity97 = smooth_conv(center_times97_1, smoothed_velocity97_1, mov_avg_win)
        
        # SG filter
        winSG97 = 80
        orderSG = 3
        SG_smooth_data97 = savgol_filter(smoothed_velocity97, winSG97, orderSG)
        
        #---
        ## calculate offset
        toff_start = -100
        toff_end = 20
        avg97 = calc_offset(center_times97, SG_smooth_data97, toff_start, toff_end)
        smoothed_velocity97_SG = SG_smooth_data97 - avg97
        
        #---
        ## plotting

        lab_97 = r'ED1 PDV Velocity #10197 - 4.8 ns tau with Boxcar window, Gaussian peakfinding, smoothed: double %.0f pt moving avg, %.0f pt SG filter' %(mov_avg_win, winSG97) # label
        # raw
        ax1.plot(t97,vel97,'-',markersize=markersize,linewidth=linewidth, label='ED1 PDV Velocity #10197 - 4.8 ns tau with Boxcar window, Gaussian peakfinding', color=(0, 0, 0))
        
        # filtered
        ax1.plot(center_times97,smoothed_velocity97_SG,'-',markersize=markersize,linewidth=linewidth, label=lab_97.format(str(tshift97)), color=(0, 1, 1))
        #---
        
        #---
        ## calculate numerical instantaneous derivative via SG
        t_avg_deriv, deriv, deriv_2 = calc_SG_deriv(center_times97, smoothed_velocity97, winSG97, orderSG)
        
        ax2.plot(t_avg_deriv, deriv, color= 'black', label='Savitzky-Golay derivative, cubic fit with %.f pt SG window' %(winSG97))
        ax3.plot(t_avg_deriv, deriv_2,color= 'black',label='Savitzky-Golay 2nd derivative, cubic fit with %.f pt SG window' %(winSG97))
        
        # call functions
        duration, t_max_comp, comp_max_vel, comp_disp, com_reg, time_center_3pt_comp_pop, vel_3pt_comp_pop = calc_parameters(t_avg_deriv, center_times97, smoothed_velocity97_SG, deriv, deriv_2)
        baseline_avg, baseline_stdev, baseline = calc_baseline_noise(center_times97, smoothed_velocity97_SG)
        t_statistic3, p_value3 = calc_stats(baseline, vel_3pt_comp_pop)
        baseline_rms_jerk, data_max_jerk, jerk_sig_noise_ratio  = calc_jerk_signal_noise(center_times97, smoothed_velocity97_SG, deriv_2)
        
        
        print('Offset (-100 to 20 ns): ', str(round(avg97,3)), 'm/s')
        print('Time between extrema of jerk (max-min):',str(round(duration,3)), 'ns')
        print('Max negative velocity: ', str(round(comp_max_vel,3)), 'm/s')
        print('Compression displacement:', str(round(-comp_disp*1e9,3)), 'nm')
        print('Baseline vel noise (-100 to 20 ns) and STDEV:', str(round(baseline_avg,5)), str(round(baseline_stdev,3)), 'm/s')
        print('T stat and P value for compression region (3 pt around max neg. vel.):', str(round(t_statistic3,3)), str(round(p_value3,3)))
        print('Baseline jerk (-100 to 20 ns) RMS: ', str(round(baseline_rms_jerk,3)))
        print('Jerk signal to noise ratio: ', str(round(jerk_sig_noise_ratio,3)))
        
        ## option to generate a second figure and plot the ROI with the velocity-history
        #plot_ROI('10197',center_times97,smoothed_velocity97_SG)
        


        
### Plotting
## plot options
ax1.set_xlim(-100, 80)
ax1.set_ylim(-10,15)

ax2.set_ylim(-15,30)
ax3.set_ylim(-10,15)

ax1.tick_params(axis='both', which='major', labelsize=8)
ax2.tick_params(axis='both', which='major', labelsize=8)
ax3.tick_params(axis='both', which='major', labelsize=8)

ax3.set_xlabel('Time [ns]', fontsize=7)
ax1.set_ylabel('Velocity [m/s]', fontsize=7)
ax2.set_ylabel('Acceleration [nm/ns$^2$]', fontsize=7)
ax3.set_ylabel('Jerk [nm/ns$^3$]', fontsize=7)

ax1.legend(fontsize=4.5, loc=2)
ax2.legend(fontsize=4.5, loc=2)
ax3.legend(fontsize=4.5, loc=2)

ax1.hlines(y = 0, xmin=-100, xmax=100,linewidth=1, color=(0,0,0), alpha=0.1)
ax2.hlines(y = 0, xmin=-100, xmax=100,linewidth=1, color=(0,0,0), alpha=0.1)
ax3.hlines(y = 0, xmin=-100, xmax=100,linewidth=1, color=(0,0,0), alpha=0.1)
           
fig.set_dpi(300)
    
    