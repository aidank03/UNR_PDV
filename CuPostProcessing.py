#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Aidanklemmer
University of Nevada, Reno
Aidanklemmer@outlook.com
4/30/24
"""

"""
Edits for Cu METI 3:
J. Iratcabal
University of Nevada, Reno
10/01/24
"""

### PDV post-processing 
#
#
#
import sys
import numpy as np
import scipy as sc
import pandas as pd
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from scipy.linalg import svd, inv
from numpy.polynomial import polynomial as P
from numpy.polynomial.polynomial import polyvander





#---
### PDV class
#
# each shot has many objects within the class
#
class PDVshot:
    
    def __init__(self, shot_num, tau): # initialize class for each shot number
    
        # initialize class object
        self.shot_num = shot_num # shot number 
        self.tau = tau # STFT window duration (tau)
        
        # add name that included info about the data
        self.data_name = []
        
        # region of interest for physics
        self.t_start = 45
        self.t_end = 72
        self.idx_start = []
        self.idx_end = []
        
        # current data
        self.ED1_current_avg = []
        self.ED1_tc_avg = []
        
        # digitizer speed (sample rate)
        # ED1 campaign used 25 GHz (0.04 ns, 40 ps, point spacing), defined as "alpha"
        self.alpha = 40e-12 # 40 ps
        
        # storage for velocity-time history data
        self.velocity_data = []
        self.velocity_data_filtered = []
        self.time_data = []
        self.velocity_unc = []
        self.baseline_vel_unc = []
        self.exp_vel_unc = []
        self.avg_baseline_vel_unc = []
        self.avg_exp_vel_unc = []
        
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
        self.min_2nd_jerk_tag = []
        
        # extrema 
        self.extrema_accel = []
        self.extrema_jerk = []
        
        # compression 
        self.max_neg_vel = []
        self.time_max_neg_vel = []
        self.compression_disp = [] # displacement from compression (integral of velocity from -20 ns until the v = 0)
        self.comp_t_stat = [] # compression t-test t-stat
        self.comp_p_val = [] # compression t-test p-value

        # times of interest
        self.time_vel_0 = [] # time of v = 0 m/s
        self.time_vel_2 = [] # time of v = 2 m/s
        self.time_vel_4 = [] # time of v = 4 m/s
    
        # time between extrema of jerk (duration of melt)
        self.duration_melt = []
        
        # baseline (reference) vel values
        self.baseline_vel = []
        self.baseline_vel_avg = []
        self.baseline_vel_stdev = []
        self.baseline_vel_rms = []
        
        # offset t-test
        self.baseline_offset_t_stat = []
        self.baseline_offset_p_val = []
        
        # baseline (reference) accel values
        self.baseline_accel = []
        self.baseline_accel_avg = []
        self.baseline_accel_stdev = []
        self.baseline_accel_rms = []
        
        # baseline (reference) jerk values
        self.baseline_jerk = []
        self.baseline_jerk_avg = []
        self.baseline_jerk_stdev = []
        self.baseline_jerk_rms = []

        # jerk signal to noise ratio
        self.jerk_SNR = [] 
        
        # surface B field
        self.surf_B_start_melt = []
        self.surf_B_end_melt = []
        
        
#---    
    
#---  
    ## class methods 
    #
    def add_name(self, name):
        self.data_name.append(name)
    
    
    # load velocity history
    def add_vel_hist(self, time, vel, unc):
        
        self.velocity_data = vel
        self.time_data = time
        self.velocity_unc = unc
    
    # load Mykonos current data
    def add_current_data(self):

        # Mykonos current data
        # new index and Bdot average data
        # tc_94_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_94_avg_100kA_60ns.txt'
        # tc_94_avg = np.loadtxt(tc_94_avg)
        # c_94_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_94_avg_100kA_60ns.txt'
        # c_94_avg = np.loadtxt(c_94_avg)

        # tc_95_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_95_avg_100kA_60ns.txt'
        # tc_95_avg = np.loadtxt(tc_95_avg)
        # c_95_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_95_avg_100kA_60ns.txt'
        # c_95_avg = np.loadtxt(c_95_avg)

        # tc_96_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_96_avg_100kA_60ns.txt'
        # tc_96_avg = np.loadtxt(tc_96_avg)
        # c_96_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_96_avg_100kA_60ns.txt'
        # c_96_avg = np.loadtxt(c_96_avg)

        # tc_97_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_97_avg_100kA_60ns.txt'
        # tc_97_avg = np.loadtxt(tc_97_avg) 
        # c_97_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_97_avg_100kA_60ns.txt'
        # c_97_avg = np.loadtxt(c_97_avg)
        
        # tc_93_avg = '/Users/Jeremy/Documents/PDV/Cu Post-Processing/IGA_METI3_100kA_60ns_trace_10190_ind_end_2650_pre_data_2000_bdot_all.csv'
        # tc_93_avg = np.loadtxt(tc_93_avg, delimiter=',',usecols=(0,1)) 
        # print(tc_93_avg.shape)
        # print(tc_93_avg)
        # c_93_avg = tc_93_avg[:,1]
        # print(c_93_avg)
      
        # tc_93_avg = np.array(tc_93_avg[:,0])
        # print(tc_93_avg)
        
        tc_89 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/METI_Data/METI_III/ScopeData/IGA_METI3_100kA_60ns_trace_10189_ind_end_2650_pre_data_2000_bdot_all.csv'
        tc_89 = np.loadtxt(tc_89, delimiter=',') 
        tc_90 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/METI_Data/METI_III/ScopeData/IGA_METI3_100kA_60ns_trace_10190_ind_end_2650_pre_data_2000_bdot_all.csv'
        tc_90 = np.loadtxt(tc_90, delimiter=',')
        tc_89_avg = np.zeros([6000,2])
        tc_90_avg = np.zeros([6000,2])
        tc_93_avg = np.zeros([6000,2])
        i=int(0)
        while i<len(tc_89_avg):
            tc_89_avg[i,0] = (tc_89[i,0]+tc_89[i,2]+tc_89[i,4]+tc_89[i,6])/4
            tc_89_avg[i,1] = (tc_89[i,1]+tc_89[i,3]+tc_89[i,5]+tc_89[i,7])/4
            tc_90_avg[i,0] = (tc_90[i,0]+tc_90[i,2]+tc_90[i,4]+tc_90[i,6])/4
            tc_90_avg[i,1] = (tc_90[i,1]+tc_90[i,3]+tc_90[i,5]+tc_90[i,7])/4
            tc_93_avg[i,0] = (tc_89_avg[i,0]+tc_90_avg[i,0])/2
            tc_93_avg[i,1] = (tc_89_avg[i,1]+tc_90_avg[i,1])/2
            i+=1
        # plt.plot(tc_89_avg[:,0],tc_89_avg[:,1],color='blue')
        # plt.plot(tc_90_avg[:,0],tc_90_avg[:,1],color='green')
        # plt.plot(tc_93_avg[:,0],tc_93_avg[:,1],color='red')
        
        
        
        # for ind in range(0,len(tc_93_avg)):
        #     current_avg = (c_94_avg[ind]+c_95_avg[ind]+c_96_avg[ind]+c_97_avg[ind])/4
        #     self.ED1_current_avg.append(current_avg)
        #     tc_avg = (tc_94_avg[ind]+tc_95_avg[ind]+tc_96_avg[ind]+tc_97_avg[ind])/4
        #     self.ED1_tc_avg.append(tc_avg) 
     
        self.ED1_tc_avg.append(tc_93_avg[:,0])
        self.ED1_current_avg.append(tc_93_avg[:,1])
        
    # moving average 
    def smooth_data(self, window_size, num_iter):
        
        if num_iter == 1:
            smoothed_vel = np.convolve(self.velocity_data, np.ones(window_size)/window_size, mode='same')
            self.velocity_data_filtered = smoothed_vel
        elif num_iter == 2:
            smoothed_vel1 = np.convolve(self.velocity_data, np.ones(window_size)/window_size, mode='same')
            smoothed_vel2 = np.convolve(smoothed_vel1, np.ones(window_size)/window_size, mode='same')
            self.velocity_data_filtered = smoothed_vel2
        else:
            print('Invalid moving average filter iteration number')
            sys.exit(1) # raise system exception and code to exit
            
    # Savitzky-Golay (SG) derivatives and filters
    def calc_SG_derivatives(self, window_size):
        # alternative
        # savgol_filter_werror(y, window_length, degree, error=None, cov=None, deriv=None)
        
        order = 3 # SG filter polynomial order
        smoothed_vel_SG = savgol_filter(self.velocity_data_filtered, window_size, order, mode='mirror', delta=self.alpha)
        #smoothed_vel_SG = savgol_filter_werror(self.velocity_data_filtered, window_size, order, self.velocity_unc, delta=self.alpha)
        acceleration = savgol_filter(self.velocity_data_filtered, window_size, order, deriv=1, delta=self.alpha)
        jerk = sc.signal.savgol_filter(self.velocity_data_filtered, window_size, order, deriv=2, delta=self.alpha)
        acceleration = acceleration * 1e-9
        jerk = jerk * 1e-18
        self.velocity_data_filtered = []
        self.velocity_data_filtered = smoothed_vel_SG
        self.acceleration_data = acceleration
        self.jerk_data = jerk
       
    def calc_baseline_noise(self):
        
        # baseline 
        tbase_start = -360
        tbase_end = 0
        base_start = np.where(self.time_data >= tbase_start)[0][0]
        base_end = np.where(self.time_data <= tbase_end)[0][-1]
        # velocity
        self.baseline_vel = self.velocity_data_filtered[base_start:base_end]
        self.baseline_vel_avg = np.mean(self.baseline_vel)
        self.baseline_vel_stdev = np.std(self.baseline_vel)
        self.baseline_vel_rms = np.sqrt((np.mean(self.baseline_vel**2)))
        # acceleration
        self.baseline_accel = self.acceleration_data[base_start:base_end]
        self.baseline_accel_avg = np.mean(self.baseline_accel)
        self.baseline_accel_stdev = np.std(self.baseline_vel)
        self.baseline_accel_rms = np.sqrt((np.mean(self.baseline_accel**2)))
        # jerk
        self.baseline_jerk = self.jerk_data[base_start:base_end]
        self.baseline_jerk_avg = np.mean(self.baseline_jerk)
        self.baseline_jerk_stdev = np.std(self.baseline_jerk)
        self.baseline_jerk_rms = np.sqrt((np.mean(self.baseline_jerk**2)))
        
    def baseline_offset_t_test(self):
        
        # baseline 
        tbase_start = -360
        tbase_end = 0
        base_start = np.where(self.time_data >= tbase_start)[0][0]
        base_end = np.where(self.time_data <= tbase_end)[0][-1]
        # velocity
        self.baseline_vel = self.velocity_data_filtered[base_start:base_end]
        self.baseline_vel_avg = np.mean(self.baseline_vel)
        # calculate t-test statistics 
        # Welch t-test
        #self.baseline_offset_t_stat, self.baseline_offset_p_val = stats.ttest_1samp(5423, 0.65)  # perform T-test
        #print('t-stat: ', self.baseline_offset_t_stat)
        #print('p-value: ', self.baseline_offset_p_val)
        #print(self.baseline_vel_avg)
        
    def avg_vel_unc(self):
        
        # define time regions
        # baseline
        tbase_start = -360
        tbase_end = 0
        base_start = np.where(self.time_data >= tbase_start)[0][0]
        base_end = np.where(self.time_data <= tbase_end)[0][-1]
        self.baseline_vel_unc = self.velocity_unc[base_start:base_end]
        self.avg_baseline_vel_unc = np.mean(self.velocity_unc[base_start:base_end])
        # experiment region
        texp_start = 0
        texp_end = 72
        exp_start = np.where(self.time_data >= texp_start)[0][0]
        exp_end = np.where(self.time_data <= texp_end)[0][-1]
        self.exp_vel_unc = self.velocity_unc[exp_start:exp_end]
        self.avg_exp_vel_unc = np.mean(self.velocity_unc[exp_start:exp_end])

    
    def calc_compression_disp(self):   
        
        # compression displacement
        tcomp_start = 20
        vel_0_idx = np.where(self.velocity_data_filtered <= 0)[0][-1]
        comp_start_idx = np.where(self.time_data >= tcomp_start)[0][0]
        comp_reg = self.velocity_data_filtered[comp_start_idx:vel_0_idx]
        self.compression_disp = - sc.integrate.simpson(comp_reg, dx=self.alpha) * 1e9 # make positive and convert to nm
    
    def calc_compression_t_test(self):
        
        # find the indices that correspond to the time range
        # time domain (x1, and x2 data) are the same for both first and second derivative
        self.idx_start = np.where(self.time_data >= self.t_start)[0][0]
        self.idx_end = np.where(self.time_data <= self.t_end)[0][-1]
        
        # calculate time of max compression
        comp_max_vel = np.min(self.velocity_data_filtered[self.idx_start:self.idx_end]) # find max neg vel, this is the min vel value in the range
        self.max_neg_vel = comp_max_vel
        ind_comp_max = np.where(self.velocity_data_filtered <= comp_max_vel)
        max_neg_vel_idx = ind_comp_max[0][0]  # index of the center value
        self.time_max_neg_vel = self.time_data[max_neg_vel_idx]
        stat_shift = 0  # 3 ns shift earlier in time, this is an arbitrary choice
        unique_pt_spacing_func_tau = int(((self.tau *1e-9) / self.alpha) + 1)
        offsets = [-unique_pt_spacing_func_tau + stat_shift,  0 + stat_shift, unique_pt_spacing_func_tau + stat_shift] # define the offsets/spacing for the 3 pt comp region population
        vel_3pt_comp_pop = self.velocity_data_filtered[[max_neg_vel_idx + offset for offset in offsets]] # calculate the 3 pt comp population
        vel_3pt_comp_pop = [round(val, 2) for val in vel_3pt_comp_pop] # round
        #time_center_3pt_comp_pop = self.time_data[0][max_neg_vel_idx] # calculate the center time of the 3 pt comp population   
        # calculate t-test statistics 
        # Welch t-test
        #self.comp_t_stat, self.comp_p_val = stats.ttest_ind(vel_3pt_comp_pop,  self.baseline_vel_avg, equal_var=False)  # perform T-test
        self.comp_t_stat, self.comp_p_val = stats.ttest_1samp(vel_3pt_comp_pop, self.baseline_vel_avg)  # perform T-test

 
    def calc_extrema(self):
        
        #---
        ## perform peakfinding and calculate extrema 
        #---
        # find zero crossings
        #zero_crossings_accel = np.where(np.diff(np.sign(self.acceleration_data)))[0] # acceleration
        #zero_crossings_jerk = np.where(np.diff(np.sign(self.jerk_data)))[0] # jerk
        
        # find extrema using scipy.signal.find_peaks
        # acceleration
        peaks_accel, _ = find_peaks(self.acceleration_data)
        valleys_accel, _ = find_peaks(-np.array(self.acceleration_data))
        self.extrema_accel = np.sort(np.concatenate((peaks_accel, valleys_accel)))
        
        # find extrema using scipy.signal.find_peaks
        # jerk
        peaks_jerk, _ = find_peaks(self.jerk_data)
        valleys_jerk, _ = find_peaks(-np.array(self.jerk_data))
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
        max_peak_time_accel = self.time_data[extrema_range_accel[np.argmax(self.acceleration_data[extrema_range_accel])]]
        # index local minimum of acceleration after peak acceleration
        #min_accel_after_max_idx = np.where(self.time_data[extrema_range_accel] > max_peak_time_accel)
        #min_accel_after_max = self.time_data[extrema_range_accel[np.argmax(self.acceleration_data[extrema_range_accel])+1]]
        # find max and min values of jerk
        max_peak_time_jerk = self.time_data[extrema_range_jerk[np.argmax(self.jerk_data[extrema_range_jerk])]]
        min_peak_time_jerk = self.time_data[extrema_range_jerk[np.argmin(self.jerk_data[extrema_range_jerk])]]
        
        #print(max_peak_time_accel)
        
        # find duration of time between extrema of jerk
        self.duration_melt = min_peak_time_jerk - max_peak_time_jerk
        
        # find max acceleration (local extrema)
        self.val_max_accel = self.acceleration_data[extrema_range_accel[np.argmax(self.acceleration_data[extrema_range_accel])]]
        #print(self.val_max_accel)
        
        '''
        fig2, (ax3) = plt.subplots(1,1,figsize=(7,5), sharex=True, sharey=True)

        plt.rcParams["font.family"] = "serif"
        
        ax3.plot(self.time_data,self.acceleration_data)
        ax3.plot(max_peak_time_accel, self.val_max_accel, 'bo')
        
        ax3.set_xlim(40,80)
        ax3.set_ylim(-15,35)
        '''
        
        # find min acceleration (local extrema)
        #self.val_min_accel = np.array(self.acceleration_data)[extrema_range_accel][min_accel_after_max_idx][0]
        # time of max acceleration (local extrema)
        self.time_max_accel = max_peak_time_accel
        # time of min acceleration (local extrema)
        #self.time_min_accel = np.array(self.time_data)[extrema_range_accel][min_accel_after_max_idx][0]
        # find max jerk (local extrema)
        self.val_max_jerk = self.jerk_data[extrema_range_jerk[np.argmax(self.jerk_data[extrema_range_jerk])]]
        # find min jerk (local extrema)
        self.val_min_jerk = self.jerk_data[extrema_range_jerk[np.argmin(self.jerk_data[extrema_range_jerk])]]
        # time of max jerk (local extrema)
        self.time_max_jerk = max_peak_time_jerk
        # time of min jerk (local extrema)
        self.time_min_jerk = min_peak_time_jerk
        
        '''
        # sometimes the local min of jerk is actually early in time, before the local max of jerk. So we could either carefully define regions of time for all parameters (I feel this is not great), or we can simply take the second local min of jerk (the 2nd min)
        if self.time_min_jerk <= self.time_max_jerk:
            self.min_2nd_jerk_tag = True
            second_min_jerk_value_idx = np.where(self.jerk_data[0][extrema_range_jerk] == np.partition(self.jerk_data[0][extrema_range_jerk], 1)[1])
            min_2nd_jerk_time = self.time_data[0][extrema_range_jerk[second_min_jerk_value_idx]]
            #print(self.duration_melt)
            self.duration_melt = []
            #print(self.duration_melt)
            self.duration_melt = float(min_2nd_jerk_time - max_peak_time_jerk)
            print(self.duration_melt)
        
        else:
            self.min_2nd_jerk_tag = False
        ''' 
        if self.data_name == ['Shot10188_4.8ns_tau_boxcar']:
            
            
            #['Shot10195_4.8ns_tau_hann']
            #['Shot10195_6.4ns_tau_hann']
            #['Shot10195_9.6ns_tau_hann']
            
            #['Shot10195_4.8ns_tau_boxcar']
            #['Shot10195_6.4ns_tau_boxcar']
            #['Shot10195_9.6ns_tau_boxcar']
            
            
            if self.time_min_jerk <= self.time_max_jerk:
                
                
                '''
                fig2, (ax5, ax6) = plt.subplots(2,1,figsize=(7,5), sharex=True, sharey=True)
    
                plt.rcParams["font.family"] = "serif"
            
                ax5.plot(self.time_data,self.jerk_data)
                ax5.plot(max_peak_time_jerk, self.val_max_jerk, 'bo')
                ax5.plot(min_peak_time_jerk, self.val_min_jerk, 'ms')
                
                ax5.set_xlim(40,80)
                ax5.set_ylim(-15,20)
    
                '''
                
                '''
                fig2, (ax3, ax4, ax5, ax6) = plt.subplots(4,1,figsize=(7,5), sharex=True, sharey=True)

                plt.rcParams["font.family"] = "serif"
                
                ax3.plot(self.time_data,self.acceleration_data)
                ax3.plot(max_peak_time_accel, self.val_max_accel, 'bo')
                #ax3.plot(min_peak_time_accel, self.val_min_accel, 'ms')
            
                ax5.plot(self.time_data,self.jerk_data)
                ax5.plot(max_peak_time_jerk, self.val_max_jerk, 'bo')
                ax5.plot(min_peak_time_jerk, self.val_min_jerk, 'ms')
                '''
                
                self.min_2nd_jerk_tag = True
                
                # Extract the jerk data for the extrema range
                extrema_jerk_values = self.jerk_data[extrema_range_jerk]  # Removed the [0]
                
                # Debugging: Check what extrema_jerk_values contains
                #print(f"extrema_jerk_values: {extrema_jerk_values}")
                
                # Find the second largest value of jerk using np.partition
                second_max_jerk_value = np.sort(extrema_jerk_values)[-2]
                #print(second_max_jerk_value)
                #print(np.max(self.jerk_data[extrema_range_jerk]))
                #print(np.min(self.jerk_data[extrema_range_jerk]))
                
                
                self.val_max_jerk = second_max_jerk_value
                
                # Find the index of this second max value in the original extrema array
                second_max_jerk_value_idx = np.where(extrema_jerk_values == second_max_jerk_value)[0][0]
                
                # Find the corresponding time for this second max jerk
                max_2nd_jerk_time = self.time_data[extrema_range_jerk][second_max_jerk_value_idx]
                
                # Calculate the duration between the second max and the min jerk peak time
                self.duration_melt = min_peak_time_jerk - max_2nd_jerk_time

                
                self.time_max_jerk = max_2nd_jerk_time
                #print(self.duration_melt)
                
                #print(self.time_max_jerk)
                #print(min_2nd_jerk_time)

                
                #ax5.set_xlim(40,80)
                #ax5.set_ylim(-15,25)
                
                #ax4.plot(self.time_data,self.acceleration_data)
                #ax6.plot(self.time_data,self.jerk_data)
                #ax6.plot(self.time_max_jerk, self.val_max_jerk, 'bo')
                #ax6.plot(self.time_min_jerk, self.val_min_jerk, 'ms')
                
                
            else:
                self.min_2nd_jerk_tag = False
                
                
        if self.data_name == ['Shot10193_4.8ns_tau_boxcar'] or ['Shot10193_4.8ns_tau_hann']:
            
            
            #['Shot10195_4.8ns_tau_hann']
            #['Shot10195_6.4ns_tau_hann']
            #['Shot10195_9.6ns_tau_hann']
            
            #['Shot10195_4.8ns_tau_boxcar']
            #['Shot10195_6.4ns_tau_boxcar']
            #['Shot10195_9.6ns_tau_boxcar']
            
            
            if self.time_min_jerk <= self.time_max_jerk:
                
                
                
                
                fig2, (ax3, ax4, ax5, ax6) = plt.subplots(4,1,figsize=(7,5), sharex=True, sharey=True)

                plt.rcParams["font.family"] = "serif"
                
                ax3.plot(self.time_data,self.acceleration_data)
                ax3.plot(max_peak_time_accel, self.val_max_accel, 'bo')
                #ax3.plot(min_peak_time_accel, self.val_min_accel, 'ms')
            
                ax5.plot(self.time_data,self.jerk_data)
                ax5.plot(max_peak_time_jerk, self.val_max_jerk, 'bo')
                ax5.plot(min_peak_time_jerk, self.val_min_jerk, 'ms')
                
                
                self.min_2nd_jerk_tag = True
                
                # Extract the jerk data for the extrema range
                extrema_jerk_values = self.jerk_data[extrema_range_jerk]  # Removed the [0]
                
                # Debugging: Check what extrema_jerk_values contains
                #print(f"extrema_jerk_values: {extrema_jerk_values}")
                
                # Find the second largest value of jerk using np.partition
                second_max_jerk_value = np.sort(extrema_jerk_values)[-2]
                #print(second_max_jerk_value)
                #print(np.max(self.jerk_data[extrema_range_jerk]))
                #print(np.min(self.jerk_data[extrema_range_jerk]))
                
                
                self.val_max_jerk = second_max_jerk_value
                
                # Find the index of this second max value in the original extrema array
                second_max_jerk_value_idx = np.where(extrema_jerk_values == second_max_jerk_value)[0][0]
                
                # Find the corresponding time for this second max jerk
                max_2nd_jerk_time = self.time_data[extrema_range_jerk][second_max_jerk_value_idx]
                
                # Calculate the duration between the second max and the min jerk peak time
                self.duration_melt = min_peak_time_jerk - max_2nd_jerk_time

                
                self.time_max_jerk = max_2nd_jerk_time
                #print(self.duration_melt)
                
                #print(self.time_max_jerk)
                #print(min_2nd_jerk_time)

                
                ax5.set_xlim(40,80)
                ax5.set_ylim(-15,25)
                
                ax4.plot(self.time_data,self.acceleration_data)
                ax6.plot(self.time_data,self.jerk_data)
                ax6.plot(self.time_max_jerk, self.val_max_jerk, 'bo')
                ax6.plot(self.time_min_jerk, self.val_min_jerk, 'ms')
                
                
            else:
                self.min_2nd_jerk_tag = False
        
        
        
        
    
    # calculate signal to noise ratio for jerk
    def calc_jerk_signal_noise(self):
        
        # calculate the displacement by integrating the velocity
        tbase_start = -360
        tbase_end = 0
        base_start_idx = np.where(self.time_data >= tbase_start)[0][0] # find index of start of baseline
        base_end_idx = np.where(self.time_data <= tbase_end)[0][-1] # find index of end of baseline
        base_jerk = np.array(self.jerk_data[base_start_idx:base_end_idx])
        
        # find extrema of jerk
        data_start = 20
        data_end = 72
        data_start_idx = np.where(self.time_data >= data_start)[0][0]
        data_end_idx = np.where(self.time_data <= data_end)[0][-1]
        data_jerk = np.array(self.jerk_data[data_start_idx:data_end_idx])
        data_max_jerk = np.max(data_jerk)

        # Ensure there are always 9000 data points
        desired_points = 9000
        actual_points = base_end_idx - base_start_idx
        extra_points = desired_points - actual_points

        if extra_points > 0:
            # Need to extend the data range
            base_end_idx += extra_points
        elif extra_points < 0:
            # Need to reduce the data range
            print('Error: invalid temporal range given')

        self.baseline_jerk = np.array(self.jerk_data[base_start_idx:base_end_idx])
        self.baseline_jerk_rms = np.sqrt(np.mean(base_jerk ** 2))  # recalculate RMS
        self.jerk_SNR = data_max_jerk / self.baseline_jerk_rms # calculate signal to noise ratio       
        
    def calc_time_vel_0_2_4(self):
        
        # time of velocity threshold 
        # v = 0 m/s, 2 m/s, and v = 4 m/s
        vel_0_idx = np.where(self.velocity_data <= 0)[0][-1]
        vel_2_idx = np.where(self.velocity_data <= 2)[0][-1]
        vel_4_idx = np.where(self.velocity_data <= 4)[0][-1]
        self.time_vel_0 = self.time_data[vel_0_idx]
        self.time_vel_2 = self.time_data[vel_2_idx]
        self.time_vel_4 = self.time_data[vel_4_idx]

    def calc_surface_B_field(self):
 
        tcomp_start = 20
        comp_start_idx = np.where(self.time_data >= tcomp_start)[0][0]
        
        melt_start_idx = np.where(self.time_data >= self.time_max_jerk)[0][0]
        melt_end_idx = np.where(self.time_data >= self.time_min_jerk)[0][0]
        
        vel_comp_to_start_melt = self.velocity_data_filtered[comp_start_idx:melt_start_idx]
        vel_comp_to_end_melt = self.velocity_data_filtered[comp_start_idx:melt_end_idx]

        melt_start_idx_current = np.where(self.ED1_tc_avg >= self.time_max_jerk)[0][0]
        melt_end_idx_current = np.where(self.ED1_tc_avg >= self.time_min_jerk)[0][0]
        #print(melt_start_idx_current)
        #print(self.time_min_jerk)
        #print(self.time_max_jerk)
        current_start_melt = self.ED1_current_avg[melt_start_idx_current] 
        current_end_melt = self.ED1_current_avg[melt_end_idx_current] 
        
        self.disp_melt_start = - sc.integrate.simpson(vel_comp_to_start_melt, dx=self.alpha) # make positive and convert to nm
        self.disp_melt_end = - sc.integrate.simpson(vel_comp_to_end_melt, dx=self.alpha)  # make positive and convert to nm

        init_radius = 400.5e-6 # avg init radius over 4 shots
        
        mu_0 = 1.256637e-6 # N*A^-2
        
        self.surf_B_start_melt = (current_start_melt*1e3 * mu_0) / (2*np.pi* (init_radius + self.disp_melt_start)) # B = I*mu / (2*pi*r)
        self.surf_B_end_melt = (current_end_melt*1e3 * mu_0) / (2*np.pi* (init_radius + self.disp_melt_end)) # B = I*mu / (2*pi*r)
        
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

data_location = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/SMASH_Copper_Velocities/'


data_names = [
               '10171_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10171_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10171_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               
               '10188_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10188_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10188_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               
               '10193_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10193_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               '10193_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv',
               
               
               '10171_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10171_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10171_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               
               '10188_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10188_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10188_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               
               '10193_100kA_60ns_noremovesin_1000_1e6_tau4_8ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10193_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv',
               '10193_100kA_60ns_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_250_width.csv'
              ]

#---------------------------------------------------------

              # ['Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_90_width.csv',
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv',
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv',
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv',
              
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_90_width.csv',
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv',
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv',
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv',
              
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_90_width.csv',
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv',
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv',
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv',
              
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_90_width.csv',
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv',
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv',
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv',
              
              
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_150_width.csv', # hann
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv', # hann
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv', # hann
              # 'Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv', # hann
                
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_150_width.csv', # hann
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv', # hann
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv', # hann
              # 'Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv', # hann
              
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_150_width.csv', # hann
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv', # hann
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv', # hann
              # 'Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv', # hann
                
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_150_width.csv', # hann
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv', # hann
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv', # hann
              # 'Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv',] # hann



data = list() # storage for PDV shot data objects - this corresponds to the list of shot data names ("data_names")
            
shot_num = [10171, 10188, 10193] # choosen shot numbers - this need to correspond to the data files stored in "data_names" above
tau_values = [4.8, 6.4, 9.6] # choosen taus - this need to correspond to the data files stored in "data_names" above
win_func = ['boxcar', 'hann']

count = 0 # counter for loops

for func in win_func:
    for number in shot_num: # loop over shot numbers
        for value in tau_values: # loop over tau values
            file = data_location + data_names[count]
            file_data = np.loadtxt(file, skiprows=30, usecols=(0,1,2))
            time = (file_data[:,0]*1e9)[0:-1-600]
            vel = file_data[:,1][0:-1-600]
            unc = file_data[:,2][0:-1-600]
            
            data.append(PDVshot(str(number),float(value))) # intialize shot number and tau
            name = 'Shot' + str(number) + '_' + str(value) + 'ns_tau_' + func
            data[count].add_name(name)
            data[count].add_vel_hist(time, vel, unc) # load velocity-history
            data[count].add_current_data()
            data[count].smooth_data(12, 1) # specify moving average window size (12 pts) and number of iterations (1)
            data[count].calc_SG_derivatives(80) # specify SG filter window size (80 pts)
            data[count].calc_baseline_noise() # calculate baseline noise params
            data[count].baseline_offset_t_test()
            data[count].avg_vel_unc()
            data[count].calc_compression_disp() # calculate compression displacement
            data[count].calc_compression_t_test() # calculate compression t-test
            data[count].calc_extrema() # calculate extrema of vel, accel, and jerk data
            data[count].calc_peaks_derivatives() # calculate time and values of extrema of accel and jerk
            data[count].calc_jerk_signal_noise() # calculate jerk signal to noise ratio
            data[count].calc_time_vel_0_2_4() # calculate time at which vel = 0 m/s, 2 m/s, and 4 m/s
            data[count].calc_surface_B_field()
            dataframe = pd.DataFrame(vars(data[count]).items(), columns=['Variable', 'Value'])
            folder_loc = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/Excel_calc_dump/'
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            pd.set_option('display.precision', 5)
            #dataframe.to_excel(folder_loc + name + "_calculated_parameters.xlsx", index=False) 
            count += 1



# #---
# ### results 
# #
# # examples of accessing and printing out values of interest for each shot
# #


# # average and standard deviation duration of melt
# # boxcar
# avg_jerk_SNR_3p2_boxcar = (data[0].jerk_SNR + data[4].jerk_SNR + data[8].jerk_SNR + data[12].jerk_SNR)/4
# avg_jerk_SNR_4p8_boxcar = (data[1].jerk_SNR + data[5].jerk_SNR + data[9].jerk_SNR + data[13].jerk_SNR)/4
# avg_jerk_SNR_6p4_boxcar = (data[2].jerk_SNR + data[6].jerk_SNR + data[10].jerk_SNR + data[14].jerk_SNR)/4
# avg_jerk_SNR_9p6_boxcar = (data[3].jerk_SNR + data[7].jerk_SNR + data[15].jerk_SNR + data[15].jerk_SNR)/4


# # hann
# avg_jerk_SNR_3p2_hann = (data[16].jerk_SNR + data[20].jerk_SNR + data[24].jerk_SNR + data[28].jerk_SNR)/4
# avg_jerk_SNR_4p8_hann = (data[17].jerk_SNR + data[21].jerk_SNR + data[25].jerk_SNR + data[29].jerk_SNR)/4
# avg_jerk_SNR_6p4_hann = (data[18].jerk_SNR + data[22].jerk_SNR + data[26].jerk_SNR + data[30].jerk_SNR)/4
# avg_jerk_SNR_9p6_hann = (data[19].jerk_SNR + data[23].jerk_SNR + data[27].jerk_SNR + data[31].jerk_SNR)/4


# print('Avg jerk_SNR with 3.2 ns tau, boxcar: ', round(avg_jerk_SNR_3p2_boxcar,2))
# print('Avg jerk_SNR with 4.8 ns tau, boxcar: ', round(avg_jerk_SNR_4p8_boxcar,2))
# print('Avg jerk_SNR with 6.4 ns tau, boxcar: ', round(avg_jerk_SNR_6p4_boxcar,2))
# print('Avg jerk_SNR with 9.6 ns tau, boxcar: ', round(avg_jerk_SNR_9p6_boxcar,2))

# print('Avg jerk_SNR with 3.2 ns tau, hann: ', round(avg_jerk_SNR_3p2_hann,2))
# print('Avg jerk_SNR with 4.8 ns tau, hann: ', round(avg_jerk_SNR_4p8_hann,2))
# print('Avg jerk_SNR with 6.4 ns tau, hann: ', round(avg_jerk_SNR_6p4_hann,2))
# print('Avg jerk_SNR with 9.6 ns tau, hann: ', round(avg_jerk_SNR_9p6_hann,2))





# # average and standard deviation duration of melt
# # boxcar
# avg_duration_melt_3p2_boxcar = (data[0].duration_melt + data[4].duration_melt + data[8].duration_melt + data[12].duration_melt)/4
# stdev_duration_melt_3p2_boxcar = np.std([data[0].duration_melt, data[4].duration_melt, data[8].duration_melt, data[12].duration_melt])

# avg_duration_melt_4p8_boxcar = (data[1].duration_melt + data[5].duration_melt + data[9].duration_melt + data[13].duration_melt)/4
# stdev_duration_melt_4p8_boxcar = np.std([data[1].duration_melt, data[5].duration_melt, data[9].duration_melt, data[13].duration_melt])

# avg_duration_melt_6p4_boxcar = (data[2].duration_melt + data[6].duration_melt + data[10].duration_melt + data[14].duration_melt)/4
# stdev_duration_melt_6p4_boxcar = np.std([data[2].duration_melt, data[6].duration_melt, data[10].duration_melt, data[14].duration_melt])

# avg_duration_melt_9p6_boxcar = (data[3].duration_melt + data[7].duration_melt + data[15].duration_melt + data[15].duration_melt)/4
# stdev_duration_melt_9p6_boxcar = np.std([data[3].duration_melt, data[7].duration_melt, data[15].duration_melt, data[15].duration_melt])

# # hann
# avg_duration_melt_3p2_hann = (data[16].duration_melt + data[20].duration_melt + data[24].duration_melt + data[28].duration_melt)/4
# stdev_duration_melt_3p2_hann = np.std([data[16].duration_melt, data[20].duration_melt, data[24].duration_melt, data[28].duration_melt])

# avg_duration_melt_4p8_hann = (data[17].duration_melt + data[21].duration_melt + data[25].duration_melt + data[29].duration_melt)/4
# stdev_duration_melt_4p8_hann = np.std([data[17].duration_melt, data[21].duration_melt, data[25].duration_melt, data[29].duration_melt])

# avg_duration_melt_6p4_hann = (data[18].duration_melt + data[22].duration_melt + data[26].duration_melt + data[30].duration_melt)/4
# stdev_duration_melt_6p4_hann = np.std([data[18].duration_melt, data[22].duration_melt, data[26].duration_melt, data[30].duration_melt])

# avg_duration_melt_9p6_hann = (data[19].duration_melt + data[23].duration_melt + data[27].duration_melt + data[31].duration_melt)/4
# stdev_duration_melt_9p6_hann = np.std([data[19].duration_melt, data[23].duration_melt, data[27].duration_melt, data[31].duration_melt])

# print('Avg duration melt with 3.2 ns tau, boxcar: ', round(avg_duration_melt_3p2_boxcar,2))
# print('Avg duration melt with 4.8 ns tau, boxcar: ', round(avg_duration_melt_4p8_boxcar,2))
# print('STDEV duration melt with 4.8 ns tau, boxcar: ', round(stdev_duration_melt_4p8_boxcar,2))
# print('Avg duration melt with 6.4 ns tau, boxcar: ', round(avg_duration_melt_6p4_boxcar,1))
# print('STDEV duration melt with 6.4 ns tau, boxcar: ', round(stdev_duration_melt_6p4_boxcar,1))
# print('Avg duration melt with 9.6 ns tau, boxcar: ', round(avg_duration_melt_9p6_boxcar,2))

# print('Avg duration melt with 3.2 ns tau, hann: ', round(avg_duration_melt_3p2_hann,2))
# print('Avg duration melt with 4.8 ns tau, hann: ', round(avg_duration_melt_4p8_hann,2))
# print('Avg duration melt with 6.4 ns tau, hann: ', round(avg_duration_melt_6p4_hann,2))
# print('Avg duration melt with 9.6 ns tau, hann: ', round(avg_duration_melt_9p6_hann,2))


# print('Velocity STDEV 10197, 6.4 ns tau, boxcar: ', data[14].baseline_vel_stdev)


# print("#94, 6.4 ns tau, boxcar, time max jerk: ", data[2].time_max_jerk)
# print("#94, 6.4 ns tau, boxcar, time min jerk: ", data[2].time_min_jerk)

# print("#95, 6.4 ns tau, boxcar, time max jerk: ", data[6].time_max_jerk)
# print("#95, 6.4 ns tau, boxcar, time min jerk: ", data[6].time_min_jerk)

# print("#96, 6.4 ns tau, boxcar, time max jerk: ", data[10].time_max_jerk)
# print("#96, 6.4 ns tau, boxcar, time min jerk: ", data[10].time_min_jerk)

# print("#97, 6.4 ns tau, boxcar, time max jerk: ", data[14].time_max_jerk)
# print("#97, 6.4 ns tau, boxcar, time min jerk: ", data[14].time_min_jerk)


# print("#97, 4.8 ns tau, boxcar, time max jerk: ", data[13].time_max_jerk)
# print("#97, 4.8 ns tau, boxcar, time min jerk: ", data[13].time_min_jerk)





# avg_exp_vel_unc_6p4_boxcar = (data[2].avg_exp_vel_unc + data[6].avg_exp_vel_unc + data[10].avg_exp_vel_unc + data[14].avg_exp_vel_unc)/4
# avg_stdev_exp_vel_6p4_boxcar = np.std([data[2].avg_exp_vel_unc, data[6].avg_exp_vel_unc, data[10].avg_exp_vel_unc, data[14].avg_exp_vel_unc])
# print('Avg exp vel unc 6.4 ns tau, boxcar: ', round(avg_exp_vel_unc_6p4_boxcar,1))
# print('Avg exp vel unc STDEV 6.4 ns tau, boxcar: ', round(avg_stdev_exp_vel_6p4_boxcar,1))

# avg_baseline_vel_unc_6p4_boxcar = (data[2].avg_baseline_vel_unc + data[6].avg_baseline_vel_unc + data[10].avg_baseline_vel_unc + data[14].avg_baseline_vel_unc)/4
# avg_stdev_baseline_vel_6p4_boxcar = np.std([data[2].avg_baseline_vel_unc, data[6].avg_baseline_vel_unc, data[10].avg_baseline_vel_unc, data[14].avg_baseline_vel_unc])
# print('Avg baseline vel unc 6.4 ns tau, boxcar: ', round(avg_baseline_vel_unc_6p4_boxcar,1))
# print('Avg baseline vel unc STDEV 6.4 ns tau, boxcar: ', round(avg_stdev_baseline_vel_6p4_boxcar,1))



# avg_max_neg_vel_6p4_boxcar = (data[2].max_neg_vel + data[6].max_neg_vel + data[10].max_neg_vel + data[14].max_neg_vel)/4
# avg_stdev_max_neg_vel_6p4_boxcar = np.std([data[2].max_neg_vel, data[6].max_neg_vel, data[10].max_neg_vel, data[14].max_neg_vel])
# print('Avg max neg vel 6.4 ns tau, boxcar: ', round(avg_max_neg_vel_6p4_boxcar,1))
# print('Avg max neg vel STDEV 6.4 ns tau, boxcar: ', round(avg_stdev_max_neg_vel_6p4_boxcar,1))


# avg_comp_disp_6p4_boxcar = (data[2].compression_disp + data[6].compression_disp + data[10].compression_disp + data[14].compression_disp)/4
# avg_stdev_comp_disp_6p4_boxcar = np.std([data[2].compression_disp, data[6].compression_disp, data[10].compression_disp, data[14].compression_disp])
# print('Avg comp disp 6.4 ns tau, boxcar: ', round(avg_comp_disp_6p4_boxcar,1))
# print('Avg comp disp STDEV 6.4 ns tau, boxcar: ', round(avg_stdev_comp_disp_6p4_boxcar,1))


# avg_ttest_t_stat_6p4_boxcar = (data[2].comp_t_stat + data[6].comp_t_stat + data[10].comp_t_stat + data[14].comp_t_stat)/4
# avg_ttest_p_val_6p4_boxcar = (data[2].comp_p_val + data[6].comp_p_val + data[10].comp_p_val + data[14].comp_p_val)/4
# print('Avg t-test t stat 6.4 ns tau, boxcar: ', avg_ttest_t_stat_6p4_boxcar)
# print('Avg t-test p value 6.4 ns tau, boxcar: ', avg_ttest_p_val_6p4_boxcar)

# avg_ttest_t_stat_3p2_boxcar = (data[0].comp_t_stat + data[4].comp_t_stat + data[8].comp_t_stat + data[12].comp_t_stat)/4
# avg_ttest_p_val_3p2_boxcar = (data[0].comp_p_val + data[4].comp_p_val + data[8].comp_p_val + data[12].comp_p_val)/4
# print('Avg t-test t stat 3.2 ns tau, boxcar: ', round(avg_ttest_t_stat_3p2_boxcar,3))
# print('Avg t-test p value 3.2 ns tau, boxcar: ', round(avg_ttest_p_val_3p2_boxcar,3))


# avg_surf_B_start_melt_6p4_boxcar = (data[2].surf_B_start_melt + data[6].surf_B_start_melt + data[10].surf_B_start_melt + data[14].surf_B_start_melt)/4
# avg_surf_B_end_melt_6p4_boxcar = (data[2].surf_B_end_melt + data[6].surf_B_end_melt + data[10].surf_B_end_melt + data[14].surf_B_end_melt)/4
# print('Avg surface B field at start of melt, for 6.4 ns tau, boxcar: ', round(avg_surf_B_start_melt_6p4_boxcar,0))
# print('Avg surface B field at end of melt, for 6.4 ns tau, boxcar: ', round(avg_surf_B_end_melt_6p4_boxcar,0))


# # comp disp
comp_disp_10171_48_boxcar = data[0].compression_disp
comp_disp_10171_64_boxcar = data[1].compression_disp
comp_disp_10171_96_boxcar = data[2].compression_disp

comp_disp_10188_48_boxcar = data[3].compression_disp
comp_disp_10188_64_boxcar = data[4].compression_disp
comp_disp_10188_96_boxcar = data[5].compression_disp

comp_disp_10193_48_boxcar = data[6].compression_disp
comp_disp_10193_64_boxcar = data[7].compression_disp
comp_disp_10193_96_boxcar = data[8].compression_disp

comp_disp_10171_48_hann = data[9].compression_disp
comp_disp_10171_64_hann = data[10].compression_disp
comp_disp_10171_96_hann = data[11].compression_disp

comp_disp_10188_48_hann = data[12].compression_disp
comp_disp_10188_64_hann = data[13].compression_disp
comp_disp_10188_96_hann = data[14].compression_disp

comp_disp_10193_48_hann = data[15].compression_disp
comp_disp_10193_64_hann = data[16].compression_disp
comp_disp_10193_96_hann = data[17].compression_disp


# # value max negative vel
max_neg_vel_10171_48_boxcar = data[0].max_neg_vel
max_neg_vel_10171_64_boxcar = data[1].max_neg_vel
max_neg_vel_10171_96_boxcar = data[2].max_neg_vel

max_neg_vel_10188_48_boxcar = data[3].max_neg_vel
max_neg_vel_10188_64_boxcar = data[4].max_neg_vel
max_neg_vel_10188_96_boxcar = data[5].max_neg_vel

max_neg_vel_10193_48_boxcar = data[6].max_neg_vel
max_neg_vel_10193_64_boxcar = data[7].max_neg_vel
max_neg_vel_10193_96_boxcar = data[8].max_neg_vel

max_neg_vel_10171_48_hann = data[9].max_neg_vel
max_neg_vel_10171_64_hann = data[10].max_neg_vel
max_neg_vel_10171_96_hann = data[11].max_neg_vel

max_neg_vel_10188_48_hann = data[12].max_neg_vel
max_neg_vel_10188_64_hann = data[13].max_neg_vel
max_neg_vel_10188_96_hann = data[14].max_neg_vel

max_neg_vel_10193_48_hann = data[15].max_neg_vel
max_neg_vel_10193_64_hann = data[16].max_neg_vel
max_neg_vel_10193_96_hann = data[17].max_neg_vel



# # max jerk
time_max_jerk_10171_48_boxcar = data[0].time_max_jerk
time_max_jerk_10171_64_boxcar = data[1].time_max_jerk
time_max_jerk_10171_96_boxcar = data[2].time_max_jerk

time_max_jerk_10188_48_boxcar = data[3].time_max_jerk
time_max_jerk_10188_64_boxcar = data[4].time_max_jerk
time_max_jerk_10188_96_boxcar = data[5].time_max_jerk

time_max_jerk_10193_48_boxcar = data[6].time_max_jerk
time_max_jerk_10193_64_boxcar = data[7].time_max_jerk
time_max_jerk_10193_96_boxcar = data[8].time_max_jerk

time_max_jerk_10171_48_hann = data[9].time_max_jerk
time_max_jerk_10171_64_hann = data[10].time_max_jerk
time_max_jerk_10171_96_hann = data[11].time_max_jerk

time_max_jerk_10188_48_hann = data[12].time_max_jerk
time_max_jerk_10188_64_hann = data[13].time_max_jerk
time_max_jerk_10188_96_hann = data[14].time_max_jerk

time_max_jerk_10193_48_hann = data[15].time_max_jerk
time_max_jerk_10193_64_hann = data[16].time_max_jerk
time_max_jerk_10193_96_hann = data[17].time_max_jerk


# # min jerk
time_min_jerk_10171_48_boxcar = data[0].time_min_jerk
time_min_jerk_10171_64_boxcar = data[1].time_min_jerk
time_min_jerk_10171_96_boxcar = data[2].time_min_jerk

time_min_jerk_10188_48_boxcar = data[3].time_min_jerk
time_min_jerk_10188_64_boxcar = data[4].time_min_jerk
time_min_jerk_10188_96_boxcar = data[5].time_min_jerk

time_min_jerk_10193_48_boxcar = data[6].time_min_jerk
time_min_jerk_10193_64_boxcar = data[7].time_min_jerk
time_min_jerk_10193_96_boxcar = data[8].time_min_jerk

time_min_jerk_10171_48_hann = data[9].time_min_jerk
time_min_jerk_10171_64_hann = data[10].time_min_jerk
time_min_jerk_10171_96_hann = data[11].time_min_jerk

time_min_jerk_10188_48_hann = data[12].time_min_jerk
time_min_jerk_10188_64_hann = data[13].time_min_jerk
time_min_jerk_10188_96_hann = data[14].time_min_jerk

time_min_jerk_10193_48_hann = data[15].time_min_jerk
time_min_jerk_10193_64_hann = data[16].time_min_jerk
time_min_jerk_10193_96_hann = data[17].time_min_jerk



# plot type selector 
plot_option = 'duration melt'




# 1 x 3 tile
fig1, (ax1, ax2, ax3) = plt.subplots(1,3,figsize=(8,3), sharex=True, sharey=True)

plt.rcParams["font.family"] = "serif"


if plot_option == 'comp disp':
    
    # compression displacement 
    
    ax1.plot(4.8, comp_disp_10171_48_boxcar, 'rs', label= 'METI3 #10171 — Boxcar',fillstyle = 'none',)
    ax1.plot(4.8, comp_disp_10171_48_hann, 'ro', label= 'METI3 #10171 — Hann',fillstyle = 'none',)
    ax1.plot(6.4, comp_disp_10171_64_boxcar, 'rs',fillstyle = 'none',)
    ax1.plot(6.4, comp_disp_10171_64_hann, 'ro',fillstyle = 'none',)
    ax1.plot(9.6, comp_disp_10171_96_boxcar, 'rs',fillstyle = 'none',)
    ax1.plot(9.6, comp_disp_10171_96_hann, 'ro',fillstyle = 'none',)
    avg_comp_71 = np.mean([comp_disp_10171_48_boxcar, comp_disp_10171_48_hann,comp_disp_10171_64_boxcar, comp_disp_10171_64_hann,comp_disp_10171_96_boxcar, comp_disp_10171_96_hann])
    yerr_71 = np.std([comp_disp_10171_48_boxcar, comp_disp_10171_48_hann,comp_disp_10171_64_boxcar, comp_disp_10171_64_hann,comp_disp_10171_96_boxcar, comp_disp_10171_96_hann])
    ax1.errorbar(7.2, avg_comp_71, yerr_71, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax2.plot(4.8, comp_disp_10188_48_boxcar, 'bs', label= 'METI3 #10188 — Boxcar',fillstyle = 'none')
    ax2.plot(4.8, comp_disp_10188_48_hann, 'bo', label= 'METI3 #10188 — Hann',fillstyle = 'none')
    ax2.plot(6.4, comp_disp_10188_64_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(6.4, comp_disp_10188_64_hann, 'bo',fillstyle = 'none')
    ax2.plot(9.6, comp_disp_10188_96_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(9.6, comp_disp_10188_96_hann, 'bo',fillstyle = 'none')
    avg_comp_88 = np.mean([comp_disp_10188_48_boxcar, comp_disp_10188_48_hann,comp_disp_10188_64_boxcar, comp_disp_10188_64_hann,comp_disp_10188_96_boxcar, comp_disp_10188_96_hann])
    yerr_88 = np.std([comp_disp_10188_48_boxcar, comp_disp_10188_48_hann,comp_disp_10188_64_boxcar, comp_disp_10188_64_hann,comp_disp_10188_96_boxcar, comp_disp_10188_96_hann])
    ax2.errorbar(7.2, avg_comp_88, yerr_88, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax3.plot(4.8, comp_disp_10193_48_boxcar, 'gs', label= 'METI3 #10193 — Boxcar',fillstyle = 'none')
    ax3.plot(4.8, comp_disp_10193_48_hann, 'go', label= 'METI3 #10193 — Hann',fillstyle = 'none')
    ax3.plot(6.4, comp_disp_10193_64_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(6.4, comp_disp_10193_64_hann, 'go',fillstyle = 'none')
    ax3.plot(9.6, comp_disp_10193_96_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(9.6, comp_disp_10193_96_hann, 'go',fillstyle = 'none')
    avg_comp_93 = np.mean([comp_disp_10193_48_boxcar, comp_disp_10193_48_hann,comp_disp_10193_64_boxcar, comp_disp_10193_64_hann,comp_disp_10193_96_boxcar, comp_disp_10193_96_hann])
    yerr_93 = np.std([comp_disp_10193_48_boxcar, comp_disp_10193_48_hann,comp_disp_10193_64_boxcar, comp_disp_10193_64_hann,comp_disp_10193_96_boxcar, comp_disp_10193_96_hann])
    ax3.errorbar(7.2, avg_comp_93, yerr_93, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    
    ax1.set_ylabel('Compression disp. [nm]')
    ax3.set_ylabel('Compression disp. [nm]')
    ax1.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax2.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax3.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax1.legend(fontsize=7, frameon=True,edgecolor='k')
    ax2.legend(fontsize=7, frameon=True,edgecolor='k')
    ax3.legend(fontsize=7, frameon=True,edgecolor='k')
    ax1.set_xticks(np.arange(4.6, 10.6, step=1))
    ax1.set_xticklabels(np.arange(4.6, 10.6, step=1))
    ax1.set_xlim(4.6,9.8)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig1.set_dpi(300)
      
    #fig1.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_comp_disp_tau_METI3_tile.pdf',format='pdf', facecolor='none',dpi=600, bbox_inches='tight')
  



if plot_option == 'max jerk':
    
    # time local max jerk (start of melt)
    
    ax1.plot(4.8, time_max_jerk_10171_48_boxcar, 'rs', label= 'METI3 #10171 — Boxcar',fillstyle = 'none')
    ax1.plot(4.8, time_max_jerk_10171_48_hann, 'ro', label= 'METI3 #10171 — Hann',fillstyle = 'none')
    ax1.plot(6.4, time_max_jerk_10171_64_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(6.4, time_max_jerk_10171_64_hann, 'ro',fillstyle = 'none')
    ax1.plot(9.6, time_max_jerk_10171_96_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(9.6, time_max_jerk_10171_96_hann, 'ro',fillstyle = 'none')
    avg_time_max_jerk_71 = np.mean([time_max_jerk_10171_48_boxcar, time_max_jerk_10171_48_hann,time_max_jerk_10171_64_boxcar, time_max_jerk_10171_64_hann, time_max_jerk_10171_96_boxcar,time_max_jerk_10171_96_hann])
    yerr_71 = np.std([time_max_jerk_10171_48_boxcar, time_max_jerk_10171_48_hann,time_max_jerk_10171_64_boxcar, time_max_jerk_10171_64_hann, time_max_jerk_10171_96_boxcar,time_max_jerk_10171_96_hann])
    ax1.errorbar(7.2, avg_time_max_jerk_71, yerr_71, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax2.plot(4.8, time_max_jerk_10188_48_boxcar, 'bs', label= 'METI3 #10188 — Boxcar',fillstyle = 'none')
    ax2.plot(4.8, time_max_jerk_10188_48_hann, 'bo', label= 'METI3 #10188 — Hann',fillstyle = 'none')
    ax2.plot(6.4, time_max_jerk_10188_64_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(6.4, time_max_jerk_10188_64_hann, 'bo',fillstyle = 'none')
    ax2.plot(9.6, time_max_jerk_10188_96_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(9.6, time_max_jerk_10188_96_hann, 'bo',fillstyle = 'none')
    avg_time_max_jerk_88 = np.mean([time_max_jerk_10188_48_boxcar, time_max_jerk_10188_48_hann,time_max_jerk_10188_64_boxcar, time_max_jerk_10188_64_hann, time_max_jerk_10188_96_boxcar,time_max_jerk_10188_96_hann])
    yerr_88 = np.std([time_max_jerk_10188_48_boxcar, time_max_jerk_10188_48_hann,time_max_jerk_10188_64_boxcar, time_max_jerk_10188_64_hann, time_max_jerk_10188_96_boxcar,time_max_jerk_10188_96_hann])
    ax2.errorbar(7.2, avg_time_max_jerk_88, yerr_88, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax3.plot(4.8, time_max_jerk_10193_48_boxcar, 'gs', label= 'METI3 #10193 — Boxcar',fillstyle = 'none')
    ax3.plot(4.8, time_max_jerk_10193_48_hann, 'go', label= 'METI3 #10193 — Hann',fillstyle = 'none')
    ax3.plot(6.4, time_max_jerk_10193_64_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(6.4, time_max_jerk_10193_64_hann, 'go',fillstyle = 'none')
    ax3.plot(9.6, time_max_jerk_10193_96_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(9.6, time_max_jerk_10193_96_hann, 'go',fillstyle = 'none')
    avg_time_max_jerk_93 = np.mean([time_max_jerk_10193_48_boxcar, time_max_jerk_10193_48_hann,time_max_jerk_10193_64_boxcar, time_max_jerk_10193_64_hann, time_max_jerk_10193_96_boxcar,time_max_jerk_10193_96_hann])
    yerr_93 = np.std([time_max_jerk_10193_48_boxcar, time_max_jerk_10193_48_hann,time_max_jerk_10193_64_boxcar, time_max_jerk_10193_64_hann, time_max_jerk_10193_96_boxcar,time_max_jerk_10193_96_hann])
    ax3.errorbar(7.2, avg_time_max_jerk_93, yerr_93, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    
    ax1.set_ylabel('Time max jerk [ns]')
    ax3.set_ylabel('Time max jerk [ns]')
    ax3.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax1.legend(fontsize=7, frameon=True,edgecolor='k')
    ax2.legend(fontsize=7, frameon=True,edgecolor='k')
    ax3.legend(fontsize=7, frameon=True,edgecolor='k')
    ax1.set_xticks(np.arange(4.6, 10.6, step=1))
    ax1.set_xticklabels(np.arange(4.6, 10.6, step=1))
    ax1.set_yticks(np.arange(56, 80, step=2))
    ax1.set_yticklabels(np.arange(56, 80, step=2))
    ax1.set_xlim(4.6,9.8)
    ax1.set_ylim(64,78)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig1.set_dpi(300)
      
    #fig1.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_time_max_jerk_tau_METI3_tile.pdf',format='pdf', facecolor='none',dpi=600, bbox_inches='tight')
  


if plot_option == 'min jerk':
    
    # time local min jerk (start of melt)
    
    ax1.plot(4.8, time_min_jerk_10171_48_boxcar, 'rs', label= 'METI3 #10171 — Boxcar',fillstyle = 'none')
    ax1.plot(4.8, time_min_jerk_10171_48_hann, 'ro', label= 'METI3 #10171 — Hann',fillstyle = 'none')
    ax1.plot(6.4, time_min_jerk_10171_64_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(6.4, time_min_jerk_10171_64_hann, 'ro',fillstyle = 'none')
    ax1.plot(9.6, time_min_jerk_10171_96_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(9.6, time_min_jerk_10171_96_hann, 'ro',fillstyle = 'none')
    avg_time_min_jerk_71 = np.mean([time_min_jerk_10171_48_boxcar, time_min_jerk_10171_48_hann,time_min_jerk_10171_64_boxcar, time_min_jerk_10171_64_hann, time_min_jerk_10171_96_boxcar,time_min_jerk_10171_96_hann])
    yerr_71 = np.std([time_min_jerk_10171_48_boxcar, time_min_jerk_10171_48_hann,time_min_jerk_10171_64_boxcar, time_min_jerk_10171_64_hann, time_min_jerk_10171_96_boxcar,time_min_jerk_10171_96_hann])
    ax1.errorbar(7.2, avg_time_min_jerk_71, yerr_71, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax2.plot(4.8, time_min_jerk_10188_48_boxcar, 'bs', label= 'METI3 #10188 — Boxcar',fillstyle = 'none')
    ax2.plot(4.8, time_min_jerk_10188_48_hann, 'bo', label= 'METI3 #10188 — Hann',fillstyle = 'none')
    ax2.plot(6.4, time_min_jerk_10188_64_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(6.4, time_min_jerk_10188_64_hann, 'bo',fillstyle = 'none')
    ax2.plot(9.6, time_min_jerk_10188_96_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(9.6, time_min_jerk_10188_96_hann, 'bo',fillstyle = 'none')
    avg_time_min_jerk_88 = np.mean([time_min_jerk_10188_48_boxcar, time_min_jerk_10188_48_hann,time_min_jerk_10188_64_boxcar, time_min_jerk_10188_64_hann, time_min_jerk_10188_96_boxcar,time_min_jerk_10188_96_hann])
    yerr_88 = np.std([time_min_jerk_10188_48_boxcar, time_min_jerk_10188_48_hann,time_min_jerk_10188_64_boxcar, time_min_jerk_10188_64_hann, time_min_jerk_10188_96_boxcar,time_min_jerk_10188_96_hann])
    ax2.errorbar(7.2, avg_time_min_jerk_88, yerr_88, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax3.plot(4.8, time_min_jerk_10193_48_boxcar, 'gs', label= 'METI3 #10193 — Boxcar',fillstyle = 'none')
    ax3.plot(4.8, time_min_jerk_10193_48_hann, 'go', label= 'METI3 #10193 — Hann',fillstyle = 'none')
    ax3.plot(6.4, time_min_jerk_10193_64_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(6.4, time_min_jerk_10193_64_hann, 'go',fillstyle = 'none')
    ax3.plot(9.6, time_min_jerk_10193_96_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(9.6, time_min_jerk_10193_96_hann, 'go',fillstyle = 'none')
    avg_time_min_jerk_93 = np.mean([time_min_jerk_10193_48_boxcar, time_min_jerk_10193_48_hann,time_min_jerk_10193_64_boxcar, time_min_jerk_10193_64_hann, time_min_jerk_10193_96_boxcar,time_min_jerk_10193_96_hann])
    yerr_93 = np.std([time_min_jerk_10193_48_boxcar, time_min_jerk_10193_48_hann,time_min_jerk_10193_64_boxcar, time_min_jerk_10193_64_hann, time_min_jerk_10193_96_boxcar,time_min_jerk_10193_96_hann])
    ax3.errorbar(7.2, avg_time_min_jerk_93, yerr_93, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    
    ax1.set_ylabel('Time min jerk [ns]')
    ax3.set_ylabel('Time min jerk [ns]')
    ax3.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax1.legend(fontsize=7, frameon=True,edgecolor='k')
    ax2.legend(fontsize=7, frameon=True,edgecolor='k')
    ax3.legend(fontsize=7, frameon=True,edgecolor='k')
    ax1.set_xticks(np.arange(4.6, 10.6, step=1))
    ax1.set_xticklabels(np.arange(4.6, 10.6, step=1))
    ax1.set_yticks(np.arange(56, 80, step=2))
    ax1.set_yticklabels(np.arange(56, 80, step=2))
    ax1.set_xlim(4.6,9.8)
    ax1.set_ylim(64,78)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig1.set_dpi(300)
      
    #fig1.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_time_min_jerk_tau_METI3_tile.pdf',format='pdf', facecolor='none',dpi=600, bbox_inches='tight')
  



if plot_option == 'duration melt':
    
    # local time min jerk (end of melt)
    
    duration_melt_10171_48_boxcar = time_min_jerk_10171_48_boxcar - time_max_jerk_10171_48_boxcar
    duration_melt_10171_48_hann = time_min_jerk_10171_48_hann - time_max_jerk_10171_48_hann
    duration_melt_10171_64_boxcar = time_min_jerk_10171_64_boxcar - time_max_jerk_10171_64_boxcar
    duration_melt_10171_64_hann = time_min_jerk_10171_64_hann - time_max_jerk_10171_64_hann
    duration_melt_10171_96_boxcar = time_min_jerk_10171_96_boxcar - time_max_jerk_10171_96_boxcar
    duration_melt_10171_96_hann = time_min_jerk_10171_96_hann - time_max_jerk_10171_96_hann
    mean_duration_melt_10171 = np.mean([duration_melt_10171_48_boxcar,duration_melt_10171_48_hann,duration_melt_10171_64_boxcar,duration_melt_10171_64_hann,duration_melt_10171_96_boxcar,duration_melt_10171_96_hann])
    stdev_duration_melt_10171 = np.std([duration_melt_10171_48_boxcar,duration_melt_10171_48_hann,duration_melt_10171_64_boxcar,duration_melt_10171_64_hann,duration_melt_10171_96_boxcar,duration_melt_10171_96_hann])
    
    
    duration_melt_10188_48_boxcar = time_min_jerk_10188_48_boxcar - time_max_jerk_10188_48_boxcar
    duration_melt_10188_48_hann = time_min_jerk_10188_48_hann - time_max_jerk_10188_48_hann
    duration_melt_10188_64_boxcar = time_min_jerk_10188_64_boxcar - time_max_jerk_10188_64_boxcar
    duration_melt_10188_64_hann = time_min_jerk_10188_64_hann - time_max_jerk_10188_64_hann
    duration_melt_10188_96_boxcar = time_min_jerk_10188_96_boxcar - time_max_jerk_10188_96_boxcar
    duration_melt_10188_96_hann = time_min_jerk_10188_96_hann - time_max_jerk_10188_96_hann
    
    
    duration_melt_10188_48_hann = time_min_jerk_10188_48_hann - time_max_jerk_10188_48_hann
    
    time_max_jerk_10188_64_hann_manual = 62.86 
    time_min_jerk_10188_64_hann_manual = 67.7
    duration_melt_10188_64_hann = time_min_jerk_10188_64_hann_manual - time_max_jerk_10188_64_hann_manual
    
    time_max_jerk_10188_96_hann_manual = 62.9
    time_min_jerk_10188_96_hann_manual = 66.82
    duration_melt_10188_96_hann = time_min_jerk_10188_96_hann_manual - time_max_jerk_10188_96_hann_manual
    
    time_max_jerk_10188_96_boxcar_manual = 60.5
    time_min_jerk_10188_96_boxcar_manual = 66.26 
    duration_melt_10188_96_boxcar = time_min_jerk_10188_96_boxcar_manual - time_max_jerk_10188_96_boxcar_manual

    
    mean_duration_melt_10188 = np.mean([duration_melt_10188_48_boxcar,duration_melt_10188_48_hann,duration_melt_10188_64_boxcar,duration_melt_10188_64_hann,duration_melt_10188_96_boxcar,duration_melt_10188_96_hann])
    stdev_duration_melt_10188 = np.std([duration_melt_10188_48_boxcar,duration_melt_10188_48_hann,duration_melt_10188_64_boxcar,duration_melt_10188_64_hann,duration_melt_10188_96_boxcar,duration_melt_10188_96_hann])
    
    
    duration_melt_10193_48_boxcar = time_min_jerk_10193_48_boxcar - time_max_jerk_10193_48_boxcar
    duration_melt_10193_48_hann = time_min_jerk_10193_48_hann - time_max_jerk_10193_48_hann
    duration_melt_10193_64_boxcar = time_min_jerk_10193_64_boxcar - time_max_jerk_10193_64_boxcar
    duration_melt_10193_64_hann = time_min_jerk_10193_64_hann - time_max_jerk_10193_64_hann
    duration_melt_10193_96_boxcar = time_min_jerk_10193_96_boxcar - time_max_jerk_10193_96_boxcar
    duration_melt_10193_96_hann = time_min_jerk_10193_96_hann - time_max_jerk_10193_96_hann
    mean_duration_melt_10193 = np.mean([duration_melt_10193_48_boxcar,duration_melt_10193_48_hann,duration_melt_10193_64_boxcar,duration_melt_10193_64_hann,duration_melt_10193_96_boxcar,duration_melt_10193_96_hann])
    stdev_duration_melt_10193 = np.std([duration_melt_10193_48_boxcar,duration_melt_10193_48_hann,duration_melt_10193_64_boxcar,duration_melt_10193_64_hann,duration_melt_10193_96_boxcar,duration_melt_10193_96_hann])
    
    

    
    
    
    ax1.plot(4.8, duration_melt_10171_48_boxcar, 'rs', label= 'METI3 #10171 — Boxcar',fillstyle = 'none')
    ax1.plot(4.8, duration_melt_10171_48_hann, 'ro', label= 'METI3 #10171 — Hann',fillstyle = 'none')
    ax1.plot(6.4, duration_melt_10171_64_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(6.4, duration_melt_10171_64_hann, 'ro',fillstyle = 'none')
    ax1.plot(9.6, duration_melt_10171_96_boxcar, 'rs',fillstyle = 'none')
    ax1.plot(9.6, duration_melt_10171_96_hann, 'ro',fillstyle = 'none')
    ax1.errorbar(7.2, mean_duration_melt_10171, stdev_duration_melt_10171, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
    
    ax2.plot(4.8, duration_melt_10188_48_boxcar, 'bs', label= 'METI3 #10188 — Boxcar',fillstyle = 'none')
    ax2.plot(4.8, duration_melt_10188_48_hann, 'bo', label= 'METI3 #10188 — Hann',fillstyle = 'none')
    ax2.plot(6.4, duration_melt_10188_64_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(6.4, duration_melt_10188_64_hann, 'bo',fillstyle = 'none')
    ax2.plot(9.6, duration_melt_10188_96_boxcar, 'bs',fillstyle = 'none')
    ax2.plot(9.6, duration_melt_10188_96_hann, 'bo',fillstyle = 'none')
    ax2.errorbar(7.2, mean_duration_melt_10188, stdev_duration_melt_10188, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
   
    ax3.plot(4.8, duration_melt_10193_48_boxcar, 'gs', label= 'METI3 #10193 — Boxcar',fillstyle = 'none')
    ax3.plot(4.8, duration_melt_10193_48_hann, 'go', label= 'METI3 #10193 — Hann',fillstyle = 'none')
    ax3.plot(6.4, duration_melt_10193_64_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(6.4, duration_melt_10193_64_hann, 'go',fillstyle = 'none')
    ax3.plot(9.6, duration_melt_10193_96_boxcar, 'gs',fillstyle = 'none')
    ax3.plot(9.6, duration_melt_10193_96_hann, 'go',fillstyle = 'none')
    ax3.errorbar(7.2, mean_duration_melt_10193, stdev_duration_melt_10193, capsize=3, markersize = 5, fmt="o", fillstyle = 'none',color='k')
   

    ax1.set_ylabel(r'Duration melt [ns]')
    ax1.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax2.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax3.set_xlabel(r'STFT duration $\tau$ [ns]')
    ax1.legend(fontsize=7, frameon=True,edgecolor='k')
    ax2.legend(fontsize=7, frameon=True,edgecolor='k')
    ax3.legend(fontsize=7, frameon=True,edgecolor='k')
    ax1.set_xticks(np.arange(4.6, 10.6, step=1))
    ax1.set_xticklabels(np.arange(4.6, 10.6, step=1))
    ax1.set_xlim(4.6,9.8)
    ax1.set_yticks(np.arange(1, 10, step=1))
    ax1.set_yticklabels(np.arange(1, 10, step=1))
    ax1.set_ylim(1,8)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig1.set_dpi(300)
      
    #fig1.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_duration_melt_tau_METI3_tile.pdf',format='pdf', facecolor='none',dpi=600, bbox_inches='tight')
  


















