#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:38:03 2023

@author: Aidanklemmer
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid.inset_locator import (InsetPosition, mark_inset)
import matplotlib.patches as patches
#import matplotlib.patheffects as pe
from scipy import signal, integrate

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'dejavusans'


def smooth_conv(time, vel, window_size):
    smoothed_vel = np.convolve(vel, np.ones(window_size)/window_size, mode='same')
    return time,smoothed_vel

def calc_offset(time, vel, toff_start, toff_end):
    # Find indices corresponding to region of interest
    i_start = np.searchsorted(time, toff_start)
    i_end = np.searchsorted(time, toff_end)
    # Calculate average over region
    avg = np.mean(vel[i_start:i_end])
    return avg

def calc_SG_deriv(time, vel, win, order):
    SG_coeffs = signal.savgol_coeffs(win, order, deriv=1, delta=0.04e-9,use='dot')
    #print('length SG coeffs', len(SG_coeffs))
    SG_coeffs_2 = signal.savgol_coeffs(win, order, deriv=2,delta=0.04e-9, use='dot')
    SG_coeffs_3 = signal.savgol_coeffs(win, order, deriv=3,delta=0.04e-9, use='dot')
    deriv_avg_vel = []
    deriv2_avg_vel = []
    deriv3_avg_vel = []
    t_avg_deriv = []
    start_loop = int(win/2)
    end_loop = int(len(vel)-win/2)
    
    for j in range(start_loop, end_loop, 1):
        vals = vel[int(j-win/2): int(j+win/2)]
        #print('length vals', len(vals))
        t_avg_deriv.append(time[j])
        deriv_avg_vel.append(SG_coeffs.dot(vals))
        deriv2_avg_vel.append(SG_coeffs_2.dot(vals))
        deriv3_avg_vel.append(SG_coeffs_3.dot(vals))
    deriv = [x*1*10**-9 for x in deriv_avg_vel]
    deriv_2 = [x*1*10**-18  for x in deriv2_avg_vel]
    #deriv_3 = [x*1 for x in deriv3_avg_vel]
    
    return t_avg_deriv, deriv, deriv_2







mode = 'melt'



markersize = 0.75
linewidth = 1.35



'''
tshift94 = 3.65
tshift95 = 4.86
tshift96 = 5.56
tshift97 = 4.62
'''

'''
tshift94 = 3.6675
tshift95 = 3.6675
tshift96 = 3.6675
tshift97 = 3.6675
'''
tshift94 = 0
tshift95 = 0
tshift96 = 0
tshift97 = 0




if 'comp' in mode:
    fig, (ax1) = plt.subplots(1,1,figsize=(8,6))
    ax1.axhline(0, color='grey', lw=0.5)
    twin1 = ax1.twinx()

elif 'melt' in mode:
    fig, (ax1, ax0) = plt.subplots(2,1,figsize=(8,7),gridspec_kw={'height_ratios': [2.5, 1]})
    #fig, (ax1) = plt.subplots(1,1,figsize=(8,6))
    ax1.axhline(0, color='grey', lw=0.5)
    twin1 = ax1.twinx()
    print('melt')
    
else:
    print('Invalid option')
    
    
    
    
    
## ---
# Mykonos current data

# new index and Bdot average data
tc_94_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_94_avg_100kA_60ns.txt'
tc_94_avg = np.loadtxt(tc_94_avg)
c_94_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_94_avg_100kA_60ns.txt'
c_94_avg = np.loadtxt(c_94_avg)

tc_95_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_95_avg_100kA_60ns.txt'
tc_95_avg = np.loadtxt(tc_95_avg)
c_95_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_95_avg_100kA_60ns.txt'
c_95_avg = np.loadtxt(c_95_avg)

tc_96_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_96_avg_100kA_60ns.txt'
tc_96_avg = np.loadtxt(tc_96_avg)
c_96_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_96_avg_100kA_60ns.txt'
c_96_avg = np.loadtxt(c_96_avg)

tc_97_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/t_97_avg_100kA_60ns.txt'
tc_97_avg = np.loadtxt(tc_97_avg) 
c_97_avg = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ScopeData/c_97_avg_100kA_60ns.txt'
c_97_avg = np.loadtxt(c_97_avg)


ED1_current_avg = []
ED1_tc_avg = []
    
for ind in range(0,len(tc_94_avg)):
    current_avg = (c_94_avg[ind]+c_95_avg[ind]+c_96_avg[ind]+c_97_avg[ind])/4
    ED1_current_avg.append(current_avg)
    tc_avg = (tc_94_avg[ind]+tc_95_avg[ind]+tc_96_avg[ind]+tc_97_avg[ind])/4
    ED1_tc_avg.append(tc_avg) 

if 'comp' in mode:      
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2)

if 'melt' in mode:
    current_line, = twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Mykonos Current', color='grey',linewidth=2)


    
    
    
    
    


file94_32_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data94_32_hann = np.loadtxt(file94_32_hann, skiprows=30, usecols=(0,1,2))
t94_32_hann = (data94_32_hann[:,0]*1e9)[0:-1-400] + tshift94
vel94_32_hann = data94_32_hann[:,1][0:-1-400] 
vel_unc_94 = data94_32_hann[:,2][0:-1-400] 

smoothwin94 = 12

winSG94 = 80
orderSG = 3

#---

# smooth data
# Calculate moving average using convolution

center_times94_32_SG1, smoothed_velocity94_32_SG1 = smooth_conv(t94_32_hann, vel94_32_hann, smoothwin94)
center_times94_32_SG, smoothed_velocity94_32_SG = smooth_conv(t94_32_hann, smoothed_velocity94_32_SG1, 1)
#lab_1_1_32 = r'ED1 PDV Velocity #10194 - 3.2 ns tau with Hann window, Gaussian peakfinding, smoothed: %.0f pt moving avg, %.0f pt SG filter' %(smoothwin94, winSG94)



lab_1_1_32 = r'Experiment'


SG_smooth_data94 = savgol_filter(smoothed_velocity94_32_SG, winSG94, orderSG)


#---
## calculate offset

toff_start = 0
toff_end = 20
avg94 = calc_offset(center_times94_32_SG, SG_smooth_data94, toff_start, toff_end)
print('Offset (0-20 ns): ', avg94)
SG_smooth_data94 = SG_smooth_data94 
#---

#ax1.plot(t97[0:4786], mov2_avg97,'-',markersize=markersize,linewidth=linewidth, label=lab_1_1_32.format(str(tshift)), color=(0, 0, 0))
exp_line1, = ax1.plot(center_times94_32_SG[0:-1-50] ,SG_smooth_data94[0:-1-50] ,'-',label= 'Experiment', markersize=markersize,linewidth=linewidth, color=(139/256, 10/256, 165/256), zorder=10)  #color='#FFC400'
#---
# alt option 1: 6.4 ns tau with max PF and hann window skip=1
#file94_64='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_max_hann_600_600_2000_2000_3000.txt'








# ----



file95_32_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data95_32_hann = np.loadtxt(file95_32_hann, skiprows=30, usecols=(0,1))
t95_32_hann = (data95_32_hann[:,0]*1e9)[0:-1-400] + tshift95
vel95_32_hann = data95_32_hann[:,1][0:-1-400]


smoothwin95 = 12

winSG95 = 80
orderSG = 3

#---
# smooth data
# Calculate moving average using convolution
center_times95_32_SG1, smoothed_velocity95_32_SG1 = smooth_conv(t95_32_hann, vel95_32_hann, smoothwin95)
center_times95_32_SG, smoothed_velocity95_32_SG = smooth_conv(t95_32_hann, smoothed_velocity95_32_SG1, 1)
#lab_1_1_32 = r'ED1 PDV Velocity #10195 - 3.2 ns tau with Hann window, Gaussian peakfinding, smoothed: %.0f pt moving avg, %0.0f pt SG filter' %(smoothwin95, winSG95)
lab_1_1_32 = r'ED1 PDV Velocity #10195'


SG_smooth_data95 = savgol_filter(smoothed_velocity95_32_SG, winSG95, orderSG)

#---
## calculate offset

toff_start = 0
toff_end = 20

avg95 = calc_offset(center_times95_32_SG, SG_smooth_data95, toff_start, toff_end)
print('Offset (0-20 ns): ', avg95)
SG_smooth_data95 = SG_smooth_data95
#---

#ax1.plot(t97[0:4786], mov2_avg97,'-',markersize=markersize,linewidth=linewidth, label=lab_1_1_32.format(str(tshift)), color=(0, 0, 0))
exp_line2, = ax1.plot(center_times95_32_SG[0:-1-50],SG_smooth_data95[0:-1-50],'-',markersize=markersize,linewidth=linewidth, color=(139/256, 10/256, 165/256), zorder=10)  #'#FF1A00'
#---


file96_32_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data96_32_hann = np.loadtxt(file96_32_hann, skiprows=30, usecols=(0,1,2))
t96_32_hann = (data96_32_hann[:,0]*1e9)[0:-1-400] + tshift96
vel96_32_hann = data96_32_hann[:,1][0:-1-400] 
error96_32_hann = data96_32_hann[:,2][0:-1-400] 

smoothwin96 = 12


winSG96 = 80
orderSG = 3


#---
# smooth data
# Calculate moving average using convolution
center_times96_32_SG1, smoothed_velocity96_32_SG1 = smooth_conv(t96_32_hann, vel96_32_hann, smoothwin96)
center_times96_32_SG, smoothed_velocity96_32_SG = smooth_conv(t96_32_hann, smoothed_velocity96_32_SG1, 1)
#lab_1_1_32 = r'ED1 PDV Velocity #10196 - 3.2 ns tau with Hann window, Gaussian peakfinding, smoothed: %.0f moving avg, %0.0f pt SG filter' %(smoothwin96, winSG96)
lab_1_1_32 = r'ED1 PDV Velocity #10196'


SG_smooth_data96 = savgol_filter(smoothed_velocity96_32_SG, winSG96, orderSG)

#---
## calculate offset

toff_start = 0
toff_end = 20
avg96 = calc_offset(center_times96_32_SG , SG_smooth_data96, toff_start, toff_end)
print('Offset (0-20 ns): ', avg96)
SG_smooth_data96 = SG_smooth_data96
#---

#ax1.plot(t97[0:4786], mov2_avg97,'-',markersize=markersize,linewidth=linewidth, label=lab_1_1_32.format(str(tshift)), color=(0, 0, 0))
exp_line3, = ax1.plot(center_times96_32_SG[0:-1-50],SG_smooth_data96[0:-1-50],'-',markersize=markersize,linewidth=linewidth, color=(139/256, 10/256, 165/256), zorder=10)  #'#00CDFF'
#---


#file97_32_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau3_2ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_new.csv'
file97_32_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
file97_96_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv'
file97_64_hann='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv'
file97_96_boxcar='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'


data97_32_hann = np.loadtxt(file97_32_hann, skiprows=30, usecols=(0,1,2))
t97_32_hann = (data97_32_hann[:,0]*1e9)[0:-1-400] + tshift97
vel97_32_hann = data97_32_hann[:,1][0:-1-400] 

data97_64_hann = np.loadtxt(file97_64_hann, skiprows=30, usecols=(0,1,2))
vel_unc_97_64 = data97_64_hann[:,2][0:-1-400] 

data97_96_hann = np.loadtxt(file97_96_hann, skiprows=30, usecols=(0,1,2))
t97_96_hann = (data97_96_hann[:,0]*1e9)[0:-1-400] + tshift97
vel97_96_hann = data97_96_hann[:,1][0:-1-400] 
vel_unc_97_96 = data97_96_hann[:,2][0:-1-400] 


data97_96_boxcar = np.loadtxt(file97_96_boxcar, skiprows=30, usecols=(0,1,2))
t97_96_boxcar = (data97_96_boxcar[:,0]*1e9)[0:-1-400] + tshift97
vel97_96_boxcar = data97_96_boxcar[:,1][0:-1-400] 
vel_unc_97_96_boxcar = data97_96_boxcar[:,2][0:-1-400] 


smoothwin97 = 12

winSG97 = 80
orderSG = 3


#---
# smooth data
# Calculate moving average using convolution
center_times97_32_SG1, smoothed_velocity97_32_SG1 = smooth_conv(t97_32_hann, vel97_32_hann, smoothwin97)
center_times97_32_SG, smoothed_velocity97_32_SG = smooth_conv(t97_32_hann, smoothed_velocity97_32_SG1, 1)
#lab_1_1_32 = r'ED1 PDV Velocity #10197 - 3.2 ns tau with Hann window, Gaussian peakfinding, smoothed: %.0f pt moving avg, %0.0f pt SG filter' %(smoothwin97, winSG97)
lab_1_1_32 = r'ED1 PDV Velocity #10197'


SG_smooth_data97 = savgol_filter(smoothed_velocity97_32_SG, winSG97, orderSG)

#---
## calculate offset

toff_start = 0
toff_end = 20   
avg97 = calc_offset(center_times97_32_SG, SG_smooth_data97, toff_start, toff_end)
print('Offset (0-20 ns): ', avg97)
SG_smooth_data97 = SG_smooth_data97 
#---

#ax1.plot(t97[0:4786], mov2_avg97,'-',markersize=markersize,linewidth=linewidth, label=lab_1_1_32.format(str(tshift)), color=(0, 0, 0))
exp_line4, = ax1.plot(center_times97_32_SG[0:-1-50],SG_smooth_data97[0:-1-50],'-',markersize=markersize,linewidth=linewidth, color=(139/256, 10/256, 165/256), zorder=10) #'#00FF22'
#---

'''
file97_1_92='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau1_92ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI.txt'
data97_1_92 = np.loadtxt(file97_1_92, skiprows=30, usecols=(0,1))
t97_1_92 = (data97_1_92[:,0]*1e9)[0:-1-400] + tshift97
vel97_1_92 = data97_1_92[:,1][0:-1-400] 
center_times97_1_92, smoothed_velocity97_1_92 = smooth_conv(t97_1_92, vel97_1_92, 100)
avg_off97_1_92 = calc_offset(center_times97_1_92, smoothed_velocity97_1_92 , toff_start, toff_end)
smoothed_velocity97_1_92 = smoothed_velocity97_1_92 - avg_off97_1_92
ax1.plot(center_times97_1_92 , smoothed_velocity97_1_92 ,':', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='black', markeredgecolor='black')  #color='#FFC400'
## ---
file97_96='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad20x_-400_-200_centroid_hann.txt'
data97_96 = np.loadtxt(file97_96, skiprows=30, usecols=(0,1))
t97_96 = (data97_96[:,0]*1e9)[0:-1-400] + tshift97
vel97_96 = data97_96[:,1][0:-1-400] 
center_times97_96, smoothed_velocity97_96 = smooth_conv(t97_96, vel97_96, 100)
avg_off97_96 = calc_offset(center_times97_96, smoothed_velocity97_96 , toff_start, toff_end)
smoothed_velocity97_96 = smoothed_velocity97_96 - avg_off97_96
ax1.plot(center_times97_96[25:-1-10], smoothed_velocity97_96[25:-1-10],'-.',label=r'$\tau = 9.6$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='black', markeredgecolor='black')  #color='#FFC400'
'''



time_avg = []
vel_avg = []
    
for ind in range(0,len(SG_smooth_data95[0:-1-100])):
    v_avg = (SG_smooth_data94[0:-1][ind]+SG_smooth_data95[0:-1][ind]+SG_smooth_data96[0:-1][ind]+SG_smooth_data97[0:-1][ind])/4
    vel_avg.append(v_avg)
    t_avg = (center_times94_32_SG[0:-1][ind]+center_times95_32_SG[0:-1][ind]+center_times96_32_SG[0:-1][ind]+center_times97_32_SG[0:-1][ind])/4
    time_avg.append(t_avg) 




#exp_line5, = ax1.plot(time_avg,vel_avg,'-',label= 'Experiment', markersize=markersize,linewidth=2.5, color='#5500FF', zorder=10) #1100FF  #5500FF


# compression displacement
tcomp_start = 20
vel_0_idx = np.where(np.array(vel_avg) <= 0)[0][-1]
comp_start_idx = np.where(np.array(time_avg) >= tcomp_start)[0][0]
comp_reg = vel_avg[comp_start_idx:vel_0_idx]
disp_avg= integrate.cumtrapz(vel_avg, dx=0.04e-9) * 1e9 # make positive and convert to nm  #43


if 'melt' in mode:
    disp_line, = ax1.plot(time_avg[0:-1], disp_avg - 12, '-^', ms=6.5,markerfacecolor='none', markevery=70, color='gray', lw=1.5)




file94_amp = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/ED1_10194_signal.txt'
amp94 = np.loadtxt(file94_amp, skiprows=0, usecols=0)
file95_amp = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/ED1_10195_signal.txt'
amp95 = np.loadtxt(file95_amp, skiprows=0, usecols=0)
file96_amp = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/ED1_10196_signal.txt'
amp96 = np.loadtxt(file96_amp, skiprows=0, usecols=0)
file97_amp = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/ED1_10197_signal.txt'
amp97 = np.loadtxt(file97_amp, skiprows=0, usecols=0)

#t97_amp = np.linspace(data97_amp[0,1], data97_amp[0,1],  + tshift97


t94_amp = np.linspace(-0.000005, 0.000005,10000)*1e9 -196.38
t95_amp = np.linspace(-0.000005, 0.000005,10000)*1e9 -192.82
t96_amp = np.linspace(-0.000005, 0.000005,10000)*1e9 -192.45
t97_amp = np.linspace(-0.000005, 0.000005,10000)*1e9 -196.27 
#t97_amp, data97_amp_avg = smooth_conv(t97_amp, data97_amp*1e4+100, 0)



#print(avg_amp97)
#print(amp97)

def moving_window_amp(data, window_size):
    amplitudes = []
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        win_max = np.max(window) 
        win_min = np.min(window)
        diff = abs(win_max-win_min)
        amp = diff/2
        
        amplitudes.append(amp)
        
    return amplitudes

# Example usage
window_size = 25  # Specify the window size

amplitudes94 = moving_window_amp(amp94, window_size)
amplitudes95 = moving_window_amp(amp95, window_size)
amplitudes96 = moving_window_amp(amp96, window_size)
amplitudes97 = moving_window_amp(amp97, window_size)
#print(amplitudes)


amp_avg_store = []
amp_t_store = []
    
for ind in range(0,len(amplitudes94)):
    amp_avg = (amplitudes94[ind]+amplitudes95[ind]+amplitudes96[ind]+amplitudes97[ind])/4
    amp_avg_store.append(amp_avg)
    
    tamp_avg = (t94_amp[ind]+t95_amp[ind]+t96_amp[ind]+t97_amp[ind])/4
    amp_t_store.append(tamp_avg)
    

scaled_amp_avg = [x*1e3 for x in amp_avg_store]
amp_t_store = [y-3.6675 for y in amp_t_store]



if 'melt' in mode:
    amp_line, = ax1.plot(amp_t_store, scaled_amp_avg,'-s',ms=6,markerfacecolor='none', markevery=5, color='gray',lw= 1.5)
















#ax1.legend(fontsize = 9, loc= 2, frameon=False)
#twin1.legend(bbox_to_anchor=(0.5255, 0.835), fontsize = 9, frameon=False)


if 'melt' in mode:
    
    '''

    ## Inset 
    loc = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRL/10194_LQ_fit_inset.txt'
    LQ_data = np.loadtxt(loc)
    L_time = LQ_data[:,0]
    Q_time = LQ_data[:,1]
    
    
    ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax1, [0.09,0.68,0.31,0.22])
    ax2.set_axes_locator(ip)
    
    #mark_inset(ax1, ax2, loc1=1, loc2=2, lw= 0.5, fc="none", ec='0.5', zorder = 3)
    '''
    
    
    '''
    linear_fit_p1 = 21.31
    linear_fit_p2 = -1298
    
    quadratic_fit_p1 = 0.8855
    quadratic_fit_p2 = -106.7
    quadratic_fit_p3 = 3286 
    
    ax2.axvline(L_time[0]+tshift, color='gray',linewidth=0.5)
    ax2.axvline(Q_time[0]+tshift, color='gray',linewidth=0.5)
    
    ax2.plot(L_time + tshift, linear_fit_p1*L_time + linear_fit_p2, '-', linewidth=3.75, color = '#E11AFF', label='Linear Fit',solid_capstyle='round')
    ax2.plot(Q_time + tshift, quadratic_fit_p1*Q_time**2 + quadratic_fit_p2*Q_time + quadratic_fit_p3, '-', linewidth=3.75, color = '#3E30FF', label='Quadratic Fit',solid_capstyle='round')
    ax2.plot(center_times94_32_SG,SG_smooth_data94,'-',linewidth=1, label='Experiment (#10194)', color='white',markerfacecolor='none',path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    '''
    
    
    #ax2.plot(center_times94_32_SG,SG_smooth_data94,'-',linewidth=1, label='Experiment (#10194)', color='white',markerfacecolor='none',path_effects=[pe.Stroke(linewidth=2, foreground='k'), pe.Normal()])
    #ax2.plot(center_times94_32_SG,SG_smooth_data94,'-',linewidth=1, label='Experiment (#10197)', color='#001AFF',markerfacecolor='none')
    
    # 6.4 ns tau
    t_avg_deriv, deriv, deriv_2 = calc_SG_deriv(center_times97_32_SG, smoothed_velocity97_32_SG, winSG97, orderSG)
    

    #t_avg_deriv_I, deriv_2_I, deriv_2_I = calc_SG_deriv(center_times94_32_SG, smoothed_velocity94_32_SG, 100, orderSG)
    
    #twin2 = ax2.twinx()
    #twin3 = ax2.twinx()

    #twin2.plot(t_avg_deriv, deriv, '-.', lw=1, color = '#E11AFF', label='Savitzky-Golay coeff. derivative, cubic fit with %.f pt SG window: smoothed velocity' %(winSG94))
    #twin2.plot(t_avg_deriv, deriv_2,'-', lw=1, color=(139/256, 10/256, 165/256),label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG94))
    #twin2.plot(t_avg_deriv_I, deriv_2_I,'-', lw=0.75, color = 'blue',label='#10194 2nd derivative - cubic fitting with %.f pt SG window' %(winSG94))
    
    '''
    #twin2.set_ylim(-0.015,0.015)
    
    ax2.set_ylim(-25,200)
    
    ax2.set_yticks(np.arange(-25, 250, step=250))
    ax2.set_yticklabels(np.arange(-25, 250, step=250))
    
    
    twin2.set_yticks(np.arange(-12, 18, step=4))
    twin2.set_yticklabels(np.arange(-12, 18, step=4))
    
    twin2.set_ylim(-5,12)
    
    #ax2.axvline(65.4, color='gray',linewidth=0.5)
    #ax2.axvline(68.4, color='gray',linewidth=0.5)
    
    #ax2.axvline(64.44, color='gray',linewidth=0.5)
    #ax2.axvline(69.327, color='gray',linewidth=0.5)
    
    
    ax1.tick_params(axis='x', labelsize=6.5)
    ax1.tick_params(axis='y', labelsize=6.5)
    ax2.tick_params(axis='x', labelsize=6.5)
    ax2.tick_params(axis='y', labelsize=6.5)
    twin2.tick_params(axis='y', labelsize=6.5)
    #twin3.tick_params(axis='y', labelsize=6.5)
    twin1.tick_params(axis='y', labelsize=6.5)
    twin1.tick_params(axis='x', labelsize=6.5)
    
    #ax2.legend(fontsize=4,loc=2, frameon=True, framealpha=1)

    
    #ax2.axhline(0, color='grey', lw=0.5)
    twin2.axhline(0, color='grey', lw=0.5)
    
    ax2.spines['top'].set_visible(False)
    
    twin2.spines['right'].set_color((139/256, 10/256, 165/256))
    #twin3.spines['right'].set_color('#3E30FF')
    twin2.tick_params(axis='y', colors=(139/256, 10/256, 165/256))
    #twin3.tick_params(axis='y', colors='#3E30FF')
    
    
    ax2.spines['right'].set_visible(False)
    
    ax2.tick_params(axis='both', which='major', pad=0.4)
    twin2.tick_params(axis='both', which='major', pad=0.4)
    #twin3.tick_params(axis='both', which='major', pad=0.5)
    #twin3.spines.right.set_position(("axes", 1.335))
    
    #twin2.spines.bottom.set_position(("axes", 1.335))
    
    #twin3.ytick
    
    ax2.yaxis.set_label_coords(-0.135, 0.5)
    #ax2.xaxis.set_label_coords(0.5, -0.2)
    twin2.yaxis.set_label_coords(1.15, 0.5)
    #twin2.xaxis.set_label_coords(0.5, -0.15)
    
    
    

    #p1=patches.FancyArrowPatch((64.44, 140), (69.327, 140),lw=0.5, arrowstyle='<->', mutation_scale=10)
    #ax2.add_patch(p1)
    
    #ax2.text(65,155, 'Melt Duration', fontsize=5.5)
    #ax2.text(65.3,120, '$\Delta$t = 4.88 ns', fontsize=5.5)
    
    #ax2.arrow(65.1, 140, 4.6, 0, head_width=7, head_length=0.5, linewidth=0.1, color='k', length_includes_head=True)
    #/ax2.arrow(69.8, 140, -4.6, 0, head_width=7, head_length=0.5, linewidth=0.1, color='k', length_includes_head=True)
    
    
    
    ax3 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    ip3 = InsetPosition(ax1, [0.09,0.4,0.31,0.22])
    ax3.set_axes_locator(ip3)
    
    twin4 = ax3.twinx()
    
    #mark_inset(ax1, ax3, loc1=3, loc2=4, lw= 0.5, fc="none", ec='0.5', zorder = 3)
    
    ax3.plot(t97_mhd, v97_mhd,'-', lw=1, color= 'gray')
    
    t_avg_deriv_MHD, deriv_MHD, deriv_2_MHD = calc_SG_deriv(t97_mhd, v97_mhd, 100, 3)
    
    twin4.plot(t_avg_deriv_MHD, deriv_2_MHD, '-', lw=1, color = (139/256, 10/256, 165/256))
    
    twin4.axhline(0, color='grey', lw=0.5)
    
    ax2.text(55.5,225, '#10197 – Experiment', fontsize=10)
    ax3.text(55.5,180, '#10197 – MHD', fontsize=10)
    
    ax2.set_xlim(55, 75)
    #ax3.set_xlim(55, 75)
    ax3.set_ylim(-25, 200)
    #twin4.set_xlim(62, 72)
    #twin4.set_ylim(-20, 10)
    
    ax3.set_xticks(np.arange(50, 80, step=5))
    ax3.set_xticklabels(np.arange(50, 80, step=5))
    ax3.set_xlim(55,75)
    
    ax2.set_xticklabels([])
    
    ax3.set_yticks(np.arange(-25, 200, step=200))
    ax3.set_yticklabels(np.arange(-25, 200, step=200))
    
    #twin4.set_yticks(np.arange(-30, 50, step=4))
    #twin4.set_yticklabels(np.arange(-30, 50, step=4))
    twin4.set_yticks(np.arange(-12, 20, step=4))
    twin4.set_yticklabels(np.arange(-12, 20, step=4))
    twin4.set_ylim(-8, 12)
    
    twin4.tick_params(axis='y', colors=(139/256, 10/256, 165/256))
    ax3.spines['top'].set_visible(False)
    #ax3.spines['bottom'].set_visible(False)
    twin4.spines['top'].set_visible(False)
    
    twin4.spines['right'].set_color((139/256, 10/256, 165/256))
    #twin3.spines['right'].set_color('#3E30FF')
    twin4.tick_params(axis='y', colors=(139/256, 10/256, 165/256))
    #twin3.tick_params(axis='y', colors='#3E30FF')
    ax3.spines['right'].set_visible(False)
    
    ax3.tick_params(axis='both', which='major', pad=0.4)
    #ax3.tick_params(axis='x', which='major', pad=0.8)
    twin4.tick_params(axis='both', which='major', pad=0.4)
    
    ax3.tick_params(axis='x', labelsize=6.5)
    ax3.tick_params(axis='y', labelsize=6.5)
    twin4.tick_params(axis='x', labelsize=6.5)
    twin4.tick_params(axis='y', labelsize=6.5)
    
    twin4.yaxis.set_label_coords(1.15, 0.5)
    '''
    
    
    
    # ----
    twin0 = ax0.twinx()
    #twin2 = ax0.twinx()
    
    # 4.8 ns tau hann
    file97_48='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv'
    data97_48 = np.loadtxt(file97_48, skiprows=30, usecols=(0,1))
    t97_48 = (data97_48[:,0]*1e9)[0:-1-400] + tshift97
    vel97_48 = data97_48[:,1][0:-1-400]
    center_times97_48, smoothed_velocity97_48 = smooth_conv(t97_48, vel97_48, 12)
    center_times97_48, smoothed_velocity97_48 = smooth_conv(center_times97_48, smoothed_velocity97_48, 1)
    SG_smooth_data97_48 = savgol_filter(smoothed_velocity97_48,80, 3)
    avg_off97_48 = calc_offset(center_times97_48, SG_smooth_data97_48, toff_start, toff_end)
    smoothed_velocity97_48 = SG_smooth_data97_48 
    
    
    # 4.8 ns tau hann
    #ax0.plot(center_times97_48, smoothed_velocity97_48,'--',label=r'$\tau = 4.8$ ns, $\alpha = 0.04$ ns',linewidth=1,color=(139/256, 10/256, 165/256))  #color='#FFC400'
    t_avg_deriv_4_8, deriv_4_8, deriv_2_4_8 = calc_SG_deriv(center_times97_48, smoothed_velocity97_48, winSG97, orderSG)
    #twin0.plot(t_avg_deriv_4_8, deriv_2_4_8,'--', lw=1, color='#1100FF',label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG97))
    #twin2.plot(t_avg_deriv_4_8, deriv_4_8,'-', lw=1, color='lightblue',label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG97))
    
 
    # 4.8 ns tau boxcar
    ax0.plot(center_times97_32_SG,SG_smooth_data97,'-',linewidth=1, label='Experiment #10197', color=(139/256, 10/256, 165/256),markerfacecolor='none')
    twin0.plot(t_avg_deriv, deriv_2,'-', lw=1, color='#1100FF',label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG97))
    
    
    
    # 6.4 ns tau hann
    file97_64='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
    data97_64 = np.loadtxt(file97_64, skiprows=30, usecols=(0,1))
    t97_64 = (data97_64[:,0]*1e9)[0:-1-400] + tshift97
    vel97_64 = data97_64[:,1][0:-1-400]
    center_times97_64, smoothed_velocity97_64 = smooth_conv(t97_64, vel97_64, 12)
    center_times97_64, smoothed_velocity97_64 = smooth_conv(center_times97_64, smoothed_velocity97_64, 1)
    SG_smooth_data97_64 = savgol_filter(smoothed_velocity97_64,80, 3)
    avg_off97_64 = calc_offset(center_times97_64, SG_smooth_data97_64, toff_start, toff_end)
    smoothed_velocity97_64 = SG_smooth_data97_64 
    
    # 6.4 ns tau hann
    ax0.plot(center_times97_64, smoothed_velocity97_64,'-.',label=r'$\tau = 6.4$ ns, $\alpha = 0.04$ ns',linewidth=1,color=(139/256, 10/256, 165/256))  #color='#FFC400'
    t_avg_deriv_4_8, deriv_4_8, deriv_2_4_8 = calc_SG_deriv(center_times97_64, smoothed_velocity97_64, winSG97, orderSG)
    twin0.plot(t_avg_deriv_4_8, deriv_2_4_8,'-.', lw=1, color='#1100FF',label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG97))
    #twin2.plot(t_avg_deriv_4_8, deriv_4_8,'-', lw=1, color='lightblue',label='#10197 2nd derivative - cubic fitting with %.f pt SG window' %(winSG97))
    
    
    
    twin0.axhline(0, color='grey', lw=0.5)
    
    
    p0=patches.FancyArrowPatch(((60.75), 120), (64.57, 120),lw=1, arrowstyle='<->', mutation_scale=12)
    ax0.add_patch(p0)
    
    
    #t = plt.text(0.5, 0.5, 'text', transform=ax.transAxes, fontsize=30)
    #t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))
    
    
    twin0.text(59.1, -5.75, 'Solid', fontsize=11,bbox=dict(facecolor='white',alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    twin0.text(61.9, -5.75, 'Solid/Liquid', fontsize=11,bbox=dict(facecolor='white', alpha=1,edgecolor='black', boxstyle='square', lw=0.5))
    twin0.text(65.75, -5.75, 'Liquid', fontsize=11, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))

    twin0.text(59.8, 11.2, r'$\frac {d \vec{a}}{dt} > 0$', fontsize=14)
    twin0.text(64.6, 11.2, r'$\frac {d \vec{a}}{dt} < 0$', fontsize=14)
    


    ax0.text(61.8, 125, 'Melt Duration', fontsize=11)
    
    #twin0.text(58.1, 9, '#10194', fontsize=11)
    
    
    # velocity uncertainty
    #ax0.fill_between(center_times94_32_SG, SG_smooth_data94-vel_unc_94, SG_smooth_data94+vel_unc_94, color=(139/256, 10/256, 165/256), alpha=0.1)
             
             
    
    twin0.vlines(60.75, -100, 100, color='grey', lw=0.5)
    twin0.vlines(64.57, -100, 100, color='grey', lw=0.5)
    
    
    
    ax0.set_yticks(np.arange(-100, 200, step=50))
    ax0.set_yticklabels(np.arange(-100, 200, step=50))
    ax0.set_ylim(-75,150)
    
    
    
    ax0.set_xticks(np.arange(56, 78, step=1))
    ax0.set_xticklabels(np.arange(56, 78, step=1))
    ax0.set_xlim(58,68)
    
    
    twin0.set_yticks(np.arange(-30, 25, step=5))
    twin0.set_yticklabels(np.arange(-30, 25, step=5))
    twin0.set_ylim(-7.5,15)
    
    
    ax0.set_ylabel(r'$\vec{v}$ [m/s]', fontsize=15)
    ax0.set_xlabel('Time ($\mathit{t}$) [ns]', fontsize=15)
    twin0.set_ylabel(r'$\mathit{d \vec{a} /dt}$ $[\mathrm{nm/ns^3}]$', fontsize=15, color='#1100FF')
    
    twin0.spines['right'].set_color('#1100FF')
    #twin3.spines['right'].set_color('#3E30FF')
    twin0.tick_params(axis='y', colors='#1100FF')
    
    
    #ax1.text(56.2, 275, '(a)',  fontsize=16)
    #twin0.text(58.1, 8, '(b)',  fontsize=16)
    
    
    
    
  
    
if 'comp' in mode:
    
    
    
    #ax2 = plt.axes([0,0,1,1])
    # Manually set the position and relative size of the inset axes within ax1
    #ip = InsetPosition(ax1, [0.25,0.675,0.4,0.15])
    #ax2.set_axes_locator(ip)
    
    #twin2 = ax2.twinx()

    
    #mark_inset(ax1, ax2, loc1=1, loc2=4, lw= 0.5, fc="none", ec='0.5')
    
    #twin2.axhline(0.05, color='grey', lw=0.5)
    
    '''
    #data_6_4_loc = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad20x_-400_-200_gaussian_hann_narrowROI_60_60_80_100_200_200_200_300_300.txt'
    data_6_4_loc = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI_new.txt'
    
    data_6_4 = np.loadtxt(data_6_4_loc, skiprows=30, usecols=(0,1,2))
    t97_64_hann = (data_6_4[:,0]*1e9)[0:-1-400] + tshift97
    vel97_64_hann = data_6_4[:,1][0:-1-400] 
    error97_64_hann = data_6_4[:,2][0:-1-400] 
    '''
    
    #center_times97_64, smoothed_velocity97_64 = smooth_conv(t97_64_hann, vel97_64_hann, 25)
    
    #ax2.plot(center_times97_64,smoothed_velocity97_64 ,'-',markersize=markersize,linewidth=linewidth, label='Experiment (#10194)', color='black')
    
    #ax2.plot(center_times94_32_SG, SG_smooth_data94 ,'-',markersize=markersize,linewidth=linewidth, label='Experiment (#10194)', color='black')

    #ax2.fill_between(center_times97_64, smoothed_velocity97_64 - error97_64_hann, smoothed_velocity97_64 + error97_64_hann)
    
    #ax2.text(3.38,-4.2, 'Avg. all shots', fontsize=6)
    
    t_stat = [-7.9468125,	-8.034831725,	-8.25472625,	-8.6094815,	-8.972753725]
    p_value = [0.019603473,	0.01935058,	0.01896945,	0.018588045,	0.01842827]
    pts = [80, 82, 84, 86, 88]
    ns = [3.2, 3.28, 3.36, 3.44, 3.52]
    
    #ax2.plot(ns, t_stat, '^', color='black', markersize=3, label='Avg. T-value',markerfacecolor='none',path_effects=[pe.Stroke(linewidth=1.1, foreground='k'), pe.Normal()])
    #twin2.plot(ns, p_value, 'o', color='gray', markersize=3, label='Avg. P-value',markerfacecolor='none',path_effects=[pe.Stroke(linewidth=1.1, foreground='k'), pe.Normal()])
    
    #lines, labels = ax2.get_legend_handles_labels()
    #lines2, labels2 = twin2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.075, 0.85),fontsize=6,frameon=False)
    #ax2.legend(bbox_to_anchor=(0.57, 1.55),fontsize=5,frameon=False)
    #twin2.legend(bbox_to_anchor=(1.2, 1.55),fontsize=5,frameon=False)
    
    #ax2.set_ylim(-10,-6)
    #twin2.set_ylim(0.016, 0.02)
    #ax2.set_xticks(np.arange(60, 90, step=5))
    #ax2.set_xticklabels(np.arange(60, 90, step=5))
    #ax2.set_xlim(40,65)
    
    #ax2.tick_params(axis='x', labelsize=6)
    #ax2.tick_params(axis='y', labelsize=6)
    #twin2.tick_params(axis='y', labelsize=6)
    
    #ax2.set_xlabel('Point separation [ns]', fontsize=5.5)
    
    
    #ax2.spines['top'].set_visible(False)
    #twin2.spines['top'].set_visible(False)
    
    
    '''
    ax2.set_ylabel('T-value', fontsize=5.5)
    twin2.set_ylabel('P-value', fontsize=5.5)
    
    ax2.yaxis.set_label_coords(-0.18, 0.5)
    ax2.xaxis.set_label_coords(0.5, -0.375)
    twin2.yaxis.set_label_coords(1.25, 0.5)
    
    ax1.tick_params(axis='x', labelsize=6.5)
    ax1.tick_params(axis='y', labelsize=6.5)
    ax2.tick_params(axis='x', labelsize=5.5)
    ax2.tick_params(axis='y', labelsize=5.5)
    twin2.tick_params(axis='y', labelsize=5.5)
    #twin3.tick_params(axis='y', labelsize=6.5)
    twin1.tick_params(axis='y', labelsize=6.5)
    twin1.tick_params(axis='x', labelsize=6.5)
    
    '''
    
    
    ax3 = plt.axes([0,0,1,1])
    ip3 = InsetPosition(ax1, [0.06,0.32,0.73,0.55])
    ax3.set_axes_locator(ip3)
    
    #mark_inset(ax1, ax3, loc1=3, loc2=1, lw= 0.5, fc="none", ec='0.5',zorder = 3)
    
    ax3.axhline(0, color='grey', lw=1)
    
    #ax3.spines['top'].set_visible(False)
    
    ax3.patch.set_alpha(0)
    
    #comp_max = np.min(vel[idc_start:idc_end])
    idc_start = np.where(center_times97_32_SG >= 40)[0][0]
    idc_end = np.where(center_times97_32_SG <= 70)[0][-1]
    ind_comp_max = np.where(SG_smooth_data97 <= np.min(SG_smooth_data97[idc_start:idc_end]))
    
    #ax3.axvline(center_times97_32_SG[int(ind_comp_max[0])-75], ymin=-4, ymax=-3.5, linewidth=1, color='k',path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()]) #'#00FF22'
    #ax3.axvline(center_times97_32_SG[int(ind_comp_max[0])+81-75], ymin=-4, ymax=-3.5, linewidth=1,color='k',path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()]) #'#00FF22'
    #ax3.axvline(center_times97_32_SG[int(ind_comp_max[0])-81-75], ymin=-4, ymax=-3.5, linewidth=1, color='k',path_effects=[pe.Stroke(linewidth=1, foreground='k'), pe.Normal()]) #'#00FF22'
    
    
    '''
    ax3.axvline(center_times97_32_SG[int(ind_comp_max[0][0])-75], ymin=0.535, ymax=0.645,linewidth=1.5, color='k') #'#00FF22'
    ax3.axvline(center_times97_32_SG[int(ind_comp_max[0][0])+81-75], ymin=0.535, ymax=0.645, linewidth=1.5,color='k') #'#00FF22'
    ax3.axvline(center_times97_32_SG[int(ind_comp_max[0][0])-81-75], ymin=0.535, ymax=0.645, linewidth=1.5, color='k') #'#00FF22'
    
    
    p2=patches.FancyArrowPatch((47.42, -1.05), (50.86, -1.05),lw=1, arrowstyle='<->', mutation_scale=12)
    ax3.add_patch(p2)
    p3=patches.FancyArrowPatch((50.65, -1.05), (54.1, -1.05),lw=1, arrowstyle='<->', mutation_scale=12)
    ax3.add_patch(p3)
    
    #ax3.text(53.3,3.1, '3.2 ns', fontsize=5)
    #ax3.text(56.7,3.1, '3.2 ns', fontsize=5)
    #ax3.text(45.2,2.2, "3-pt Welch's $\it{t}$-test", fontsize=4.5)
    
    ax3.text(48.8,-0.8, '$\gamma$', fontsize=14)
    ax3.text(52.1,-0.8, '$\gamma$', fontsize=14)
    
    
    ax3.text(center_times97_32_SG[int(ind_comp_max[0][0])-75]-0.25,-0.65, '2', fontsize=12)
    ax3.text(center_times97_32_SG[int(ind_comp_max[0][0])+81-75]-0.25,-0.65, '3', fontsize=12)
    ax3.text(center_times97_32_SG[int(ind_comp_max[0][0])-81-75]-0.25,-0.65, '1', fontsize=12)
    
    '''
    
    
    
    #ax3.text(30.65,-4.3, '#10197', fontsize=11)
    
    
    ax3.text(41.5,1.65, 'Radial Compression', fontsize=10)
    #ax3.text(41.75,-2.9, 'Solid', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    #ax3.text(52,-3.5, 'Solid', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    
    p2=patches.FancyArrowPatch(((34), 1.5), (58, 1.5),lw=1, arrowstyle='<->', mutation_scale=12)
    ax3.add_patch(p2)
    
    #ax3.vlines(64, -10, 10, color='grey', lw=0.5)
    
    ax1.vlines(64, -100, 100, color='grey', lw=0.5)


    ax1.text(40,-7, 'Solid', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    ax1.text(65.5,-7, 'Solid/Liquid', fontsize=12,rotation=90, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    
    
    ## baseline vel STDEV for 10197 9.6 ns tau, boxcar
    # baseline_vel_stdev = 0.193 m/s
    
    vel_sigma = 0.193
    sigma_3 = 3 * vel_sigma
    sigma_5 = 5 * vel_sigma
    sigma_10 = 10 * vel_sigma
    sigma_15 = 15 * vel_sigma 
    
    #ax1.axhline(-vel_sigma, color='grey',linestyle='--', lw=1 )
    #ax1.axhline(-sigma_5, linestyle='-.', color='grey', lw=1)
    #ax1.axhline(-sigma_10, linestyle=':', color='grey', lw=1)
    
    #ax3.axhline(-sigma_3, color='grey',linestyle='--', lw=1 )
    ax3.axhline(-sigma_5, linestyle=':', color='grey', lw=1)
    ax3.axhline(-sigma_10, linestyle=':', color='grey', lw=1)
    ax3.axhline(-sigma_15, linestyle=':', color='grey', lw=1)
    
    #ax3.text(31,-0.65, r'$3\sigma$', fontsize=12)
    ax3.text(31,-1.4, r'$5\sigma$', fontsize=12)
    ax3.text(30.5,-2.4, r'$10\sigma$', fontsize=12)
    ax3.text(30.5,-3.4, r'$15\sigma$', fontsize=12)
    
    
    
    
    

    # Multiple options for compression 
    '''
    file97_1_92='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau1_92ns_alpha40ps_zeropad100x_-400_-200_gauss_hann_hist_ROI.txt'
    data97_1_92 = np.loadtxt(file97_1_92, skiprows=30, usecols=(0,1))
    t97_1_92 = (data97_1_92[:,0]*1e9)[0:-1-400] + tshift97
    vel97_1_92 = data97_1_92[:,1][0:-1-400] 
    center_times97_1_92, smoothed_velocity97_1_92 = smooth_conv(t97_1_92, vel97_1_92, 100)
    avg_off97_1_92 = calc_offset(center_times97_1_92, smoothed_velocity97_1_92 , toff_start, toff_end)
    smoothed_velocity97_1_92 = smoothed_velocity97_1_92 - avg_off97_1_92
    ax3.plot(center_times97_1_92 , smoothed_velocity97_1_92 ,':',label=r'$\tau = 1.92$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='black', markeredgecolor='black')  #color='#FFC400'
    

    #ax3.plot(center_times94_32_SG[0:-1-100],SG_smooth_data94[0:-1-100],'-',markersize=markersize,linewidth=linewidth, color='black') #'#00FF22'
    #ax3.plot(center_times95_32_SG[0:-1-100],SG_smooth_data95[0:-1-100],'-',markersize=markersize,linewidth=linewidth, color='black') #'#00FF22'
    #ax3.plot(center_times96_32_SG[0:-1-100],SG_smooth_data96[0:-1-100],'-',markersize=markersize,linewidth=linewidth, color='black') #'#00FF22'
    ax3.plot(center_times97_32_SG[0:-1-100],SG_smooth_data97[0:-1-100],'-', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='black') #'#00FF22'
    

    file97_64='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6_4ns_alpha40ps_zeropad20x_-400_-200_gaussian_hann_narrowROI_60_60_80_100_200_200_200_300_300.txt'
    data97_64 = np.loadtxt(file97_64, skiprows=30, usecols=(0,1))
    t97_64 = (data97_64[:,0]*1e9)[0:-1-400] + tshift97
    vel97_64 = data97_64[:,1][0:-1-400]
    center_times97_64, smoothed_velocity97_64 = smooth_conv(t97_64, vel97_64, 100)
    avg_off97_64 = calc_offset(center_times97_64, smoothed_velocity97_64 , toff_start, toff_end)
    smoothed_velocity97_64 = smoothed_velocity97_64 - avg_off97_64
    ax3.plot(center_times97_64 , smoothed_velocity97_64 ,'--',label=r'$\tau = 6.4$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='black', markeredgecolor='black')  #color='#FFC400'


    file97_96='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_noremovesin_1000_1e6_tau9_6ns_alpha40ps_zeropad20x_-400_-200_centroid_hann.txt'
    data97_96 = np.loadtxt(file97_96, skiprows=30, usecols=(0,1))
    t97_96 = (data97_96[:,0]*1e9)[0:-1-400] + tshift97
    vel97_96 = data97_96[:,1][0:-1-400] 
    center_times97_96_1, smoothed_velocity97_96_1 = smooth_conv(t97_96, vel97_96, 11)
    center_times97_96, smoothed_velocity97_96 = smooth_conv(center_times97_96_1, smoothed_velocity97_96_1 , 100)
    
    avg_off97_96 = calc_offset(center_times97_96, smoothed_velocity97_96 , toff_start, toff_end)
    smoothed_velocity97_96 = smoothed_velocity97_96 - avg_off97_96
    ax3.plot(center_times97_96[25:-1-10], smoothed_velocity97_96[25:-1-10],'-.',label=r'$\tau = 9.6$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='black', markeredgecolor='black')  #color='#FFC400'
    
    
    file97_big47 ='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_1_100bins_1.6_2_10x.txt'
    data97_big47 = np.loadtxt(file97_big47, skiprows=29, usecols=(0,1))
    t97_big47 = (data97_big47[:,0]*1e9) + tshift97
    vel97_big47 = data97_big47[:,1]
    center_times97_big47, smoothed_velocity97_big47 = smooth_conv(t97_big47, vel97_big47, 3)
    avg_off97_big47 = calc_offset(t97_big47, smoothed_velocity97_big47 , toff_start, toff_end)
    smoothed_velocity97_big47 = smoothed_velocity97_big47 - avg_off97_big47
    ax3.plot(center_times97_big47, smoothed_velocity97_big47,'o',label=r'$\tau = 4.76$ ns, $\alpha = 1.6$ ns',markeredgewidth=1, markersize=6,markerfacecolor="none",linewidth=1.5, color='black', markeredgecolor='black')  #color='#FFC400'
    '''
    
    # 3.2 ns tau
    #ax3.plot(center_times97_32_SG[0:-1-100],SG_smooth_data97[0:-1-100],':', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='#8900FF') #'#00FF22'
    #ax3.plot(center_times94_32_SG[0:-1-100],SG_smooth_data94[0:-1-100],'-', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='#1100FF') #'#00FF22'
    #ax3.plot(center_times95_32_SG[0:-1-100],SG_smooth_data95[0:-1-100],'-', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='#1100FF') #'#00FF22'
    #ax3.plot(center_times96_32_SG[0:-1-100],SG_smooth_data96[0:-1-100],'-', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='#1100FF') #'#00FF22'
    #ax3.plot(center_times97_32_SG[0:-1-100],SG_smooth_data97[0:-1-100],'.', label=r'$\tau = 3.2$ ns, $\alpha = 0.04$ ns',markersize=markersize,linewidth=1.5, color='blue') #'#00FF22'
    
    
    
    # 4.8 ns tau
    file97_48='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau3p2ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_150_width.csv'
    data97_48 = np.loadtxt(file97_48, skiprows=30, usecols=(0,1))
    t97_48 = (data97_48[:,0]*1e9)[0:-1-400] + tshift97
    vel97_48 = data97_48[:,1][0:-1-400]
    center_times97_48, smoothed_velocity97_48 = smooth_conv(t97_48, vel97_48, 12)
    center_times97_48, smoothed_velocity97_48 = smooth_conv(center_times97_48, smoothed_velocity97_48, 1)
    SG_smooth_data97_48 = savgol_filter(smoothed_velocity97_48,80, 3)
    avg_off97_48 = calc_offset(center_times97_48, SG_smooth_data97_48, toff_start, toff_end)
    smoothed_velocity97_48 = SG_smooth_data97_48 
    #ax3.plot(center_times97_48, smoothed_velocity97_48,'-.',label=r'$\tau = 4.8$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='#6F00FF', markeredgecolor='black')  #color='#FFC400'

    
    # 6.4 ns tau
    file97_64='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
    data97_64 = np.loadtxt(file97_64, skiprows=30, usecols=(0,1,2))
    t97_64 = (data97_64[:,0]*1e9)[0:-1-400] + tshift97
    vel97_64 = data97_64[:,1][0:-1-400]
    vel_unc_97_64 = data97_64[:,2][0:-1-400]
    center_times97_64, smoothed_velocity97_64 = smooth_conv(t97_64, vel97_64, 12)
    center_times97_64, smoothed_velocity97_64 = smooth_conv(center_times97_64, smoothed_velocity97_64, 1)
    SG_smooth_data97_64 = savgol_filter(smoothed_velocity97_64, 80, 3)
    avg_off97_64 = calc_offset(center_times97_64, SG_smooth_data97_64 , toff_start, toff_end)
    smoothed_velocity97_64 = SG_smooth_data97_64 
    ax3.plot(center_times97_64 , smoothed_velocity97_64 ,'--',label=r'$\tau = 6.4$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    #ax3.plot(center_times97_64 , smoothed_velocity97_64 ,'-',label=r'$\tau = 6.4$ ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='#1100FF', markeredgecolor='black')  #color='#FFC400'

    
    # 9.6 ns tau boxcar
    file97_96_boxcar ='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'
    data97_96_boxcar = np.loadtxt(file97_96_boxcar, skiprows=30, usecols=(0,1,2))
    t97_96_boxcar = (data97_96_boxcar[:,0]*1e9)[0:-1-400] + tshift97
    vel97_96_boxcar = data97_96_boxcar[:,1][0:-1-400]
    vel_unc_97_96_boxcar = data97_96_boxcar[:,2][0:-1-400]
    center_times97_96_boxcar, smoothed_velocity97_96_boxcar = smooth_conv(t97_96_boxcar, vel97_96_boxcar, 12)
    center_times97_96_boxcar, smoothed_velocity97_96_boxcar = smooth_conv(center_times97_96_boxcar, smoothed_velocity97_96_boxcar, 1)
    SG_smooth_data97_96_boxcar = savgol_filter(smoothed_velocity97_96_boxcar, 80, 3)
    avg_off97_96_boxcar = calc_offset(center_times97_96_boxcar, SG_smooth_data97_96_boxcar , toff_start, toff_end)
    smoothed_velocity97_96_boxcar = SG_smooth_data97_96_boxcar 
    ax3.plot(center_times97_96_boxcar , smoothed_velocity97_96_boxcar ,'-',label=r'$\tau = 9.6 ns, $\alpha = 0.04$ ns', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


    #ax3.legend(bbox_to_anchor=(0., 0.55), ncol=2, fontsize=10,  frameon=False)
    
    
    # velocity uncertainty
    #ax3.fill_between(center_times97_96, SG_smooth_data97_96-vel_unc_97_96, SG_smooth_data97_96+vel_unc_97_96, color=(139/256, 10/256, 165/256), alpha=0.1)
    
    ax3.fill_between(center_times97_64, SG_smooth_data97_64-vel_unc_97_64, SG_smooth_data97_64+vel_unc_97_64, color=(139/256, 10/256, 165/256), alpha=0.1)
    
    #ax3.fill_between(center_times97_96, smoothed_velocity97_96-vel_unc_97_96, smoothed_velocity97_96+vel_unc_97_96, color=(139/256, 10/256, 165/256), alpha=0.1)
    
    ax3.fill_between(center_times97_96_boxcar, SG_smooth_data97_96_boxcar-vel_unc_97_96_boxcar, SG_smooth_data97_96_boxcar+vel_unc_97_96_boxcar, color=(139/256, 10/256, 165/256), alpha=0.1)
    
    
    
    #center_times97_32_SG[0:-1-50],SG_smooth_data97[0:-1-50]
    
    
    
    
    
    ax3.set_ylim(-5.2,2.2)
    ax3.set_xlim(30,60)
    
    ax3.set_xlabel('$\mathit{t}$ [ns]', fontsize=14)
    #ax3.set_ylabel('$\mathit{v}$ (m/s)', fontsize=10)
    
    
    ax3.tick_params(axis='x', labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.tick_params(axis='both', which='major', pad=2,size=2)
    ax3.tick_params(axis='both', which='major', pad=2,size=2)
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    ax3.xaxis.set_label_coords(0.5, -0.1)
    
    plt.rcParams['axes.linewidth'] = 0.55
    
    

# plot MHD data

mhd_tshift94 = -0.9
mhd_tshift95 = 0.3
mhd_tshift96 = 0.7
mhd_tshift97 = -0.4


file94_mhd= '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/FLAG_MHD/1D_barewire_1.0-0.01um_ED1_10194_23716_93722.txt'
data94_mhd = np.loadtxt(file94_mhd, skiprows=7, usecols=(0,2))
t94_mhd = data94_mhd[:,0]*1e3 + mhd_tshift94
v94_mhd = data94_mhd[:,1]*1e4

file95_mhd= '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/FLAG_MHD/1D_barewire_1.0-0.01um_ED1_10195_23716_93722.txt'
data95_mhd = np.loadtxt(file95_mhd, skiprows=7, usecols=(0,2))
t95_mhd = data95_mhd[:,0]*1e3 + mhd_tshift95
v95_mhd = data95_mhd[:,1]*1e4

file96_mhd= '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/FLAG_MHD/1D_barewire_1.0-0.01um_ED1_10196_23716_93722.txt'
data96_mhd = np.loadtxt(file96_mhd, skiprows=7, usecols=(0,2))
t96_mhd = data96_mhd[:,0]*1e3 + mhd_tshift96
v96_mhd = data96_mhd[:,1]*1e4

file97_mhd= '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/FLAG_MHD/1D_barewire_1.0-0.01um_ED1_10197_23716_93722.txt'
data97_mhd = np.loadtxt(file97_mhd, skiprows=7, usecols=(0,2))
t97_mhd = data97_mhd[:,0]*1e3 + mhd_tshift97
v97_mhd = data97_mhd[:,1]*1e4






if 'comp' in mode:
    vshift = 0
    
    # nice purple color
    #color= '#990099'
    
    
    ax1.plot(t94_mhd, v94_mhd + vshift,'--',lw=linewidth, label= 'MHD Calculation', color= 'k')
    ax1.plot(t95_mhd, v95_mhd + vshift,'--', lw=linewidth, color= 'k')
    ax1.plot(t96_mhd, v96_mhd + vshift,'--', lw=linewidth, color= 'k')
    ax1.plot(t97_mhd, v97_mhd + vshift,'--', lw=linewidth, color= 'k')
    
    
if 'melt' in mode:
    vshift = 0
    
    mhd_line1, = ax1.plot(t94_mhd, v94_mhd + vshift,'--', markevery=125, ms=8, lw=linewidth, label= 'MHD Calculation', color= 'k', zorder=9)
    mhd_line2, = ax1.plot(t95_mhd, v95_mhd + vshift,'--', markevery=125,ms=8, lw=linewidth, color= 'k', zorder=9)
    mhd_line3, = ax1.plot(t96_mhd, v96_mhd + vshift,'--', markevery=125,ms=8, lw=linewidth, color= 'k', zorder=9)
    mhd_line4, = ax1.plot(t97_mhd, v97_mhd + vshift,'--', markevery=125,ms=8, lw=linewidth, color= 'k', zorder=9)  







if 'comp' in mode:
    
    
    ax1.set_xlim(20,68)
    ax1.set_ylim(-10,50)
    twin1.set_ylim(-50,250)
    
    ax1.set_xlabel('Time ($\mathit{t}$) [ns]', fontsize=15)
    ax1.set_ylabel(r'Velocity ($\mathit{\vec{v}}$) [m/s]', fontsize=15)

    ax1.yaxis.set_label_coords(-0.06, 0.5)

    twin1.set_ylabel('Current [kA]', fontsize=15)

    #ax1.spines['top'].set_visible(False)
    #ax1.spines['right'].set_visible(False)
    #twin1.spines['top'].set_visible(False)
    
    ax1.legend(bbox_to_anchor=(0.64, 0.98), fontsize = 11.7,ncol=2, frameon=False)
    twin1.legend(bbox_to_anchor=(0.87, 0.98), fontsize = 11.7, frameon=False)
    
    ax1.tick_params(axis='both', which='major', pad=4,size=2,labelsize=14)
    #ax2.tick_params(axis='both', which='major', pad=4,size=2)
    twin1.tick_params(axis='both', which='major', pad=4,size=2,labelsize=14)
    #twin2.tick_params(axis='both', which='major', pad=4,size=2)
    
    plt.rcParams['axes.linewidth'] = 0.55
    
    plt.rcParams["font.family"] = "serif"
    
    #plt.title('ED1 all shots with hann windowing, gaussian PF, tau=4.8ns, double half-ns move avg, and 80pt SG filter')
    #plt.title('ED1 all shots with hann windowing, gaussian PF, tau=6.4ns, no filter')

    
if 'melt' in mode:
    
    
    
    ax1.set_xticks(np.arange(52, 78, step=2))
    ax1.set_xticklabels(np.arange(52, 78, step=2))
    ax1.set_xlim(54,72)
    
    
    ax1.set_ylim(-55, 250)
    twin1.set_ylim(-110,500)
    
    #ax1.set_xlabel('Time ($\mathit{t}$) [ns]', fontsize=15)
    ax1.set_ylabel(r'$\mathit{\vec{v}}$ [m/s], $\Delta \mathrm{\vec{s}}$ [nm], A [mV]', fontsize=15)
    twin1.set_ylabel('Current [kA]', fontsize=15,)
    
    #ax2.set_ylabel('$\mathit{v}$ $\mathrm{[m/s]}$', fontsize=12)
    #ax3.set_ylabel('$\mathit{v}$ $\mathrm{[m/s]}$', fontsize=12)
    #ax2.yaxis.set_label_coords(-0.1, 0.45)
    #ax3.yaxis.set_label_coords(-0.1 , 0.45)
    
    #ax1.spines['top'].set_visible(False)
    #ax1.spines['right'].set_visible(False)
    #twin1.spines['top'].set_visible(False)
    #twin2.spines['top'].set_visible(False)
    #twin1.spines['right'].set_visible(False)
    
    
    #ax2.vlines(60.5, -100, 300, color='grey', lw=0.5)
    

    
    
    # Custom legend on ax2 (twin subplot)
    twin1.legend([disp_line, amp_line, current_line], [ r'Displacement ($\Delta \mathrm{\vec{s}}$)',r'Signal Amplitude ($A$)','Current'],loc=(0.04, 0.45), fontsize = 12, frameon=True)

    ax1.legend(fontsize = 12, loc=(0.2, 1), ncol=2, frameon=False)
    #twin1.legend(bbox_to_anchor=(0.98, 0.35), fontsize = 12, frameon=False)
    
    #ax3.set_xlabel('$\mathit{t}$ $[ns]$', fontsize=13)
    #twin2.set_ylabel(r'$\mathit{\vec a}$ $[\mathrm{m/s^2}$]', fontsize=6,color='#E11AFF')
    #[$\mathrm{m/s^2}$
    #twin2.set_ylabel(r'$\mathit{d a /dt}$ $[\mathrm{nm/ns^3}]$', fontsize=11, color=(139/256, 10/256, 165/256))
    #twin4.set_ylabel(r'$\mathit{d a /dt}$ $[\mathrm{nm/ns^3}]$', fontsize=11, color=(139/256, 10/256, 165/256))
    
    #[\mathrm{m/s^2}$]
    
    #ax2.xaxis.set_label_coords(0.5, -0.2)
    ax1.yaxis.set_label_coords(-0.09, 0.5)
    
    twin1.yaxis.set_label_coords(1.1, 0.5)
    #twin4.yaxis.set_label_coords(1.14 , 0.45)
    

    
    
    #plt.tight_layout()
    ax0.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    twin0.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    ax1.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    #ax2.tick_params(axis='both', which='major', pad=2,size=1, labelsize=12)
    twin1.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    #twin2.tick_params(axis='both', which='major', pad=2,size=1, labelsize=12)
    
    #ax3.tick_params(axis='both', which='major', pad=2,size=1, labelsize=12)
    #ax3.tick_params(axis='x', which='major', pad=4,size=1, labelsize=12)
    #twin4.tick_params(axis='both', which='major', pad=2,size=1, labelsize=12)
    
    
    plt.rcParams['axes.linewidth'] = 0.55
    #ax1.tick_params(left = False,bottom = False)
    #twin1.tick_params(right = False)
    
    plt.rcParams["font.family"] = "serif"
    
    plt.subplots_adjust(wspace=0.1, hspace=0.13)
    


#font = {'weight' : 'normal'}  
#plt.rc('font', **font) 

'''
twin1.spines["right"].set_linewidth(0.5)
twin1.spines["bottom"].set_linewidth(0.5)
ax1.spines["bottom"].set_linewidth(0.5)
ax1.spines["left"].set_linewidth(0.5)
twin1.spines["left"].set_linewidth(0.5)
ax2.spines["left"].set_linewidth(0.5)
ax2.spines["bottom"].set_linewidth(0.5)
ax2.spines["right"].set_linewidth(0.5)
ax2.spines["top"].set_linewidth(0.5)

ax3.spines["left"].set_linewidth(0.5)
ax3.spines["bottom"].set_linewidth(0.5)
ax3.spines["right"].set_linewidth(0.5)
ax3.spines["top"].set_linewidth(0.5)

twin2.spines["top"].set_linewidth(0.5)
'''


fig.set_dpi(300)
#fig.set_dpi(800) # 800 for figures at 8x6 in


#fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_comp_fig2.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')

#fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_melt_fig3_amp_disp.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight',transparent=False)




