#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 19 13:11:49 2024

@author: Aidanklemmer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid.inset_locator import (InsetPosition, mark_inset)
import matplotlib.patches as patches
#import matplotlib.patheffects as pe
from scipy import signal, integrate
from scipy.signal import find_peaks

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




mode = 'ED1 PDV amplitude'


if 'ED1 all' in mode:
 
    fig, (ax1) = plt.subplots(1,1,figsize=(8,6))
    ax1.axhline(0, color='grey', lw=0.75)
    twin1 = ax1.twinx()

if 'ED1 tau window comparison' in mode:
 
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(6,6))
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)
    twin1 = ax1.twinx()
    twin2 = ax2.twinx()
    
if 'ED1 tiled' in mode:

 
    fig, axes  = plt.subplots(2,3,figsize=(9,4))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)
    ax3.axhline(0, color='grey', lw=0.75)
    ax4.axhline(0, color='grey', lw=0.75)
    ax5.axhline(0, color='grey', lw=0.75)
    ax6.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax2.twinx()
    twin3 = ax3.twinx()
    twin4 = ax4.twinx()
    twin5 = ax5.twinx()
    twin6 = ax6.twinx()
    
if 'ED1 comp' in mode:
    
    fig, axes  = plt.subplots(2,1,figsize=(5,7),gridspec_kw={'height_ratios': [1, 1.5]})
    ax1, ax2 = axes.flatten()
    
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax2.twinx()

if 'ED1 vel/accel/jerk' in mode:
    
    fig, axes  = plt.subplots(2,1,figsize=(5,7),gridspec_kw={'height_ratios': [1, 1.5]})
    ax1, ax2 = axes.flatten()
    
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax1.twinx()
    
    twin3 = ax2.twinx()
    
if 'ED1 MHD' in mode:
    
    fig, axes  = plt.subplots(2,1,figsize=(5,7),gridspec_kw={'height_ratios': [1, 1]})
    ax1, ax2 = axes.flatten()
    
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)


    twin1 = ax1.twinx()
    twin2 = ax2.twinx()
    
    
if 'MHD v/a/j' in mode:
    
    fig, ax1  = plt.subplots(1,1,figsize=(5,6))
    
    ax1.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax1.twinx()
    twin3 = ax1.twinx()
    
if 'ED1 PDV MHD melt' in mode:
    
    fig, axes  = plt.subplots(2,1,figsize=(5,7), gridspec_kw={'height_ratios': [1, 1]})
    
    ax1, ax2 = axes.flatten()
    
    ax1.axhline(0, color='grey', lw=0.75)
    ax2.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax2.twinx()
    
if 'ED1 PDV amplitude' in mode:
    
    fig, ax1  = plt.subplots(1,1,figsize=(7,5))
    
    ax1.axhline(0, color='grey', lw=0.75)
    
    twin1 = ax1.twinx()
    twin2 = ax1.twinx()
    
    

    
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

if 'ED1 all' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2.5,zorder=1)

if 'ED1 tau window comparison' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2.5,zorder=1)
    twin2.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2.5,zorder=1)

if 'ED1 tiled' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin2.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin3.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin4.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin5.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin6.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)

if 'ED1 comp' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2,zorder=1)
    twin2.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2,zorder=1)

if 'ED1 MHD' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)
    twin2.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)

if 'ED1 PDV MHD melt' in mode:
    twin1.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=1.5,zorder=1)

if 'ED1 PDV amplitude' in mode:
    twin2.plot(ED1_tc_avg, ED1_current_avg, ':',label='Current', color='grey',linewidth=2,zorder=1)




### Hann

# 4.8 ns tau

file94_48 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv'
data94_48 = np.loadtxt(file94_48, skiprows=30, usecols=(0,1))
t94_48 = (data94_48[:,0]*1e9)[0:-1-400] 
vel94_48 = data94_48[:,1][0:-1-400]
center_times94_48, smoothed_velocity94_48 = smooth_conv(t94_48, vel94_48, 12)
center_times94_48, smoothed_velocity94_48 = smooth_conv(center_times94_48, smoothed_velocity94_48, 1)
SG_smooth_data94_48 = savgol_filter(smoothed_velocity94_48,80, 3)
smoothed_velocity94_48 = SG_smooth_data94_48 
#ax1.plot(center_times94_48[0:-1-55], smoothed_velocity94_48[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_48 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv'
data95_48 = np.loadtxt(file95_48, skiprows=30, usecols=(0,1))
t95_48 = (data95_48[:,0]*1e9)[0:-1-400] 
vel95_48 = data95_48[:,1][0:-1-400]
center_times95_48, smoothed_velocity95_48 = smooth_conv(t95_48, vel95_48, 12)
center_times95_48, smoothed_velocity95_48 = smooth_conv(center_times95_48, smoothed_velocity95_48, 1)
SG_smooth_data95_48 = savgol_filter(smoothed_velocity95_48,80, 3)
smoothed_velocity95_48 = SG_smooth_data95_48 
#ax1.plot(center_times95_48[0:-1-55], smoothed_velocity95_48[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_48 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv'
data96_48 = np.loadtxt(file96_48, skiprows=30, usecols=(0,1))
t96_48 = (data96_48[:,0]*1e9)[0:-1-400] 
vel96_48 = data96_48[:,1][0:-1-400]
center_times96_48, smoothed_velocity96_48 = smooth_conv(t96_48, vel96_48, 12)
center_times96_48, smoothed_velocity96_48 = smooth_conv(center_times96_48, smoothed_velocity96_48, 1)
SG_smooth_data96_48 = savgol_filter(smoothed_velocity96_48,80, 3)
smoothed_velocity96_48 = SG_smooth_data96_48 
#ax1.plot(center_times96_48[0:-1-55], smoothed_velocity96_48[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_48 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_100_width.csv'
data97_48 = np.loadtxt(file97_48, skiprows=30, usecols=(0,1))
t97_48 = (data97_48[:,0]*1e9)[0:-1-400] 
vel97_48 = data97_48[:,1][0:-1-400]
center_times97_48, smoothed_velocity97_48 = smooth_conv(t97_48, vel97_48, 12)
center_times97_48, smoothed_velocity97_48 = smooth_conv(center_times97_48, smoothed_velocity97_48, 1)
SG_smooth_data97_48 = savgol_filter(smoothed_velocity97_48,80, 3)
smoothed_velocity97_48 = SG_smooth_data97_48 
#ax1.plot(center_times97_48[0:-1-55], smoothed_velocity97_48[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


# 6.4 ns tau

file94_64 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv'
data94_64 = np.loadtxt(file94_64, skiprows=30, usecols=(0,1))
t94_64 = (data94_64[:,0]*1e9)[0:-1-290] 
vel94_64 = data94_64[:,1][0:-1-290]
center_times94_64, smoothed_velocity94_64 = smooth_conv(t94_64, vel94_64, 12)
center_times94_64, smoothed_velocity94_64 = smooth_conv(center_times94_64, smoothed_velocity94_64, 1)
SG_smooth_data94_64 = savgol_filter(smoothed_velocity94_64,80, 3)
smoothed_velocity94_64 = SG_smooth_data94_64 
#ax1.plot(center_times94_64[0:-1-55], smoothed_velocity94_64[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_64 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv'
data95_64 = np.loadtxt(file95_64, skiprows=30, usecols=(0,1))
t95_64 = (data95_64[:,0]*1e9)[0:-1-290] 
vel95_64 = data95_64[:,1][0:-1-290]
center_times95_64, smoothed_velocity95_64 = smooth_conv(t95_64, vel95_64, 12)
center_times95_64, smoothed_velocity95_64 = smooth_conv(center_times95_64, smoothed_velocity95_64, 1)
SG_smooth_data95_64 = savgol_filter(smoothed_velocity95_64,80, 3)
smoothed_velocity95_64 = SG_smooth_data95_64 
#ax1.plot(center_times95_64[0:-1-55], smoothed_velocity95_64[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_64 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv'
data96_64 = np.loadtxt(file96_64, skiprows=30, usecols=(0,1))
t96_64 = (data96_64[:,0]*1e9)[0:-1-290] 
vel96_64 = data96_64[:,1][0:-1-290]
center_times96_64, smoothed_velocity96_64 = smooth_conv(t96_64, vel96_64, 12)
center_times96_64, smoothed_velocity96_64 = smooth_conv(center_times96_64, smoothed_velocity96_64, 1)
SG_smooth_data96_64 = savgol_filter(smoothed_velocity96_64,80, 3)
smoothed_velocity96_64 = SG_smooth_data96_64 
#ax1.plot(center_times96_64[0:-1-55], smoothed_velocity96_64[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_64 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_75_width.csv'
data97_64 = np.loadtxt(file97_64, skiprows=30, usecols=(0,1,2))
t97_64 = (data97_64[:,0]*1e9)[0:-1-290] 
vel97_64 = data97_64[:,1][0:-1-290]
vel97_unc_64 = data97_64[:,2][0:-1-290]
center_times97_64, smoothed_velocity97_64 = smooth_conv(t97_64, vel97_64, 12)
center_times97_64, smoothed_velocity97_64 = smooth_conv(center_times97_64, smoothed_velocity97_64, 1)
SG_smooth_data97_64 = savgol_filter(smoothed_velocity97_64,80, 3)
smoothed_velocity97_64 = SG_smooth_data97_64 
#ax1.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


# 9.6 ns tau

file94_96 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv'
data94_96 = np.loadtxt(file94_96, skiprows=30, usecols=(0,1))
t94_96 = (data94_96[:,0]*1e9)[0:-1-290] 
vel94_96 = data94_96[:,1][0:-1-290]
center_times94_96, smoothed_velocity94_96 = smooth_conv(t94_96, vel94_96, 12)
center_times94_96, smoothed_velocity94_96 = smooth_conv(center_times94_96, smoothed_velocity94_96, 1)
SG_smooth_data94_96 = savgol_filter(smoothed_velocity94_96,80, 3)
smoothed_velocity94_96 = SG_smooth_data94_96 
#ax1.plot(center_times94_96[0:-1-55], smoothed_velocity94_96[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_96 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv'
data95_96 = np.loadtxt(file95_96, skiprows=30, usecols=(0,1))
t95_96 = (data95_96[:,0]*1e9)[0:-1-290] 
vel95_96 = data95_96[:,1][0:-1-290]
center_times95_96, smoothed_velocity95_96 = smooth_conv(t95_96, vel95_96, 12)
center_times95_96, smoothed_velocity95_96 = smooth_conv(center_times95_96, smoothed_velocity95_96, 1)
SG_smooth_data95_96 = savgol_filter(smoothed_velocity95_96,80, 3)
smoothed_velocity95_96 = SG_smooth_data95_96 
#ax1.plot(center_times95_96[0:-1-55], smoothed_velocity95_96[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_96 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv'
data96_96 = np.loadtxt(file96_96, skiprows=30, usecols=(0,1))
t96_96 = (data96_96[:,0]*1e9)[0:-1-290] 
vel96_96 = data96_96[:,1][0:-1-290]
center_times96_96, smoothed_velocity96_96 = smooth_conv(t96_96, vel96_96, 12)
center_times96_96, smoothed_velocity96_96 = smooth_conv(center_times96_96, smoothed_velocity96_96, 1)
SG_smooth_data96_96 = savgol_filter(smoothed_velocity96_96,80, 3)
smoothed_velocity96_96 = SG_smooth_data96_96 
#ax1.plot(center_times96_96[0:-1-55], smoothed_velocity96_96[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_96 = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_hann_hist_ROI_50_width.csv'
data97_96 = np.loadtxt(file97_96, skiprows=30, usecols=(0,1,2))
t97_96 = (data97_96[:,0]*1e9)[0:-1-290] 
vel97_96 = data97_96[:,1][0:-1-290]
vel97_unc_96 = data97_96[:,2][0:-1-290]
center_times97_96, smoothed_velocity97_96 = smooth_conv(t97_96, vel97_96, 12)
center_times97_96, smoothed_velocity97_96 = smooth_conv(center_times97_96, smoothed_velocity97_96, 1)
SG_smooth_data97_96 = savgol_filter(smoothed_velocity97_96,80, 3)
smoothed_velocity97_96 = SG_smooth_data97_96 
#ax1.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

t_SG_smooth_data97_96, deriv_97_96, deriv2_97_96 = calc_SG_deriv(center_times97_96, smoothed_velocity97_96, 80, 3)


### Boxcar

# 4.8 ns tau

file94_48_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data94_48_boxcar = np.loadtxt(file94_48_boxcar, skiprows=30, usecols=(0,1))
t94_48_boxcar = (data94_48_boxcar[:,0]*1e9)[0:-1-350] 
vel94_48_boxcar = data94_48_boxcar[:,1][0:-1-350]
center_times94_48_boxcar, smoothed_velocity94_48_boxcar = smooth_conv(t94_48_boxcar, vel94_48_boxcar, 12)
center_times94_48_boxcar, smoothed_velocity94_48_boxcar = smooth_conv(center_times94_48_boxcar, smoothed_velocity94_48_boxcar, 1)
SG_smooth_data94_48_boxcar = savgol_filter(smoothed_velocity94_48_boxcar,80, 3)
smoothed_velocity94_48_boxcar = SG_smooth_data94_48_boxcar
#ax1.plot(center_times94_48[0:-1-55], smoothed_velocity94_48[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_48_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data95_48_boxcar = np.loadtxt(file95_48_boxcar, skiprows=30, usecols=(0,1))
t95_48_boxcar = (data95_48_boxcar[:,0]*1e9)[0:-1-350] 
vel95_48_boxcar = data95_48_boxcar[:,1][0:-1-350]
center_times95_48_boxcar, smoothed_velocity95_48_boxcar = smooth_conv(t95_48_boxcar, vel95_48_boxcar, 12)
center_times95_48_boxcar, smoothed_velocity95_48_boxcar = smooth_conv(center_times95_48_boxcar, smoothed_velocity95_48_boxcar, 1)
SG_smooth_data95_48_boxcar = savgol_filter(smoothed_velocity95_48_boxcar,80, 3)
smoothed_velocity95_48_boxcar = SG_smooth_data95_48_boxcar 
#ax1.plot(center_times95_48[0:-1-55], smoothed_velocity95_48[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_48_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data96_48_boxcar = np.loadtxt(file96_48_boxcar, skiprows=30, usecols=(0,1))
t96_48_boxcar = (data96_48_boxcar[:,0]*1e9)[0:-1-350] 
vel96_48_boxcar = data96_48_boxcar[:,1][0:-1-350]
center_times96_48_boxcar, smoothed_velocity96_48_boxcar = smooth_conv(t96_48_boxcar, vel96_48_boxcar, 12)
center_times96_48_boxcar, smoothed_velocity96_48_boxcar = smooth_conv(center_times96_48_boxcar, smoothed_velocity96_48_boxcar, 1)
SG_smooth_data96_48_boxcar = savgol_filter(smoothed_velocity96_48_boxcar,80, 3)
smoothed_velocity96_48_boxcar = SG_smooth_data96_48_boxcar 
#ax1.plot(center_times96_48[0:-1-55], smoothed_velocity96_48[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_48_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau4p8ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_60_width.csv'
data97_48_boxcar = np.loadtxt(file97_48_boxcar, skiprows=30, usecols=(0,1))
t97_48_boxcar = (data97_48_boxcar[:,0]*1e9)[0:-1-350] 
vel97_48_boxcar = data97_48_boxcar[:,1][0:-1-350]
center_times97_48_boxcar, smoothed_velocity97_48_boxcar = smooth_conv(t97_48_boxcar, vel97_48_boxcar, 12)
center_times97_48_boxcar, smoothed_velocity97_48_boxcar = smooth_conv(center_times97_48_boxcar, smoothed_velocity97_48_boxcar, 1)
SG_smooth_data97_48_boxcar = savgol_filter(smoothed_velocity97_48_boxcar,80, 3)
smoothed_velocity97_48_boxcar = SG_smooth_data97_48_boxcar
#ax1.plot(center_times97_48[0:-1-55], smoothed_velocity97_48[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


# 6.4 ns tau

file94_64_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
data94_64_boxcar = np.loadtxt(file94_64_boxcar, skiprows=30, usecols=(0,1))
t94_64_boxcar = (data94_64_boxcar[:,0]*1e9)[0:-1-290] 
vel94_64_boxcar = data94_64_boxcar[:,1][0:-1-290]
center_times94_64_boxcar, smoothed_velocity94_64_boxcar = smooth_conv(t94_64_boxcar, vel94_64_boxcar, 12)
center_times94_64_boxcar, smoothed_velocity94_64_boxcar = smooth_conv(center_times94_64_boxcar, smoothed_velocity94_64_boxcar, 1)
SG_smooth_data94_64_boxcar = savgol_filter(smoothed_velocity94_64_boxcar,80, 3)
smoothed_velocity94_64_boxcar = SG_smooth_data94_64_boxcar
#ax1.plot(center_times94_64[0:-1-55], smoothed_velocity94_64[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_64_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
data95_64_boxcar = np.loadtxt(file95_64_boxcar, skiprows=30, usecols=(0,1))
t95_64_boxcar = (data95_64_boxcar[:,0]*1e9)[0:-1-290] 
vel95_64_boxcar = data95_64_boxcar[:,1][0:-1-290]
center_times95_64_boxcar, smoothed_velocity95_64_boxcar = smooth_conv(t95_64_boxcar, vel95_64_boxcar, 12)
center_times95_64_boxcar, smoothed_velocity95_64_boxcar = smooth_conv(center_times95_64_boxcar, smoothed_velocity95_64_boxcar, 1)
SG_smooth_data95_64_boxcar = savgol_filter(smoothed_velocity95_64_boxcar,80, 3)
smoothed_velocity95_64_boxcar = SG_smooth_data95_64_boxcar 
#ax1.plot(center_times95_64[0:-1-55], smoothed_velocity95_64[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_64_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
data96_64_boxcar = np.loadtxt(file96_64_boxcar, skiprows=30, usecols=(0,1))
t96_64_boxcar = (data96_64_boxcar[:,0]*1e9)[0:-1-290] 
vel96_64_boxcar = data96_64_boxcar[:,1][0:-1-290]
center_times96_64_boxcar, smoothed_velocity96_64_boxcar = smooth_conv(t96_64_boxcar, vel96_64_boxcar, 12)
center_times96_64_boxcar, smoothed_velocity96_64_boxcar = smooth_conv(center_times96_64_boxcar, smoothed_velocity96_64_boxcar, 1)
SG_smooth_data96_64_boxcar = savgol_filter(smoothed_velocity96_64_boxcar,80, 3)
smoothed_velocity96_64_boxcar = SG_smooth_data96_64_boxcar 
#ax1.plot(center_times96_64[0:-1-55], smoothed_velocity96_64[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_64_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau6p4ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_45_width.csv'
data97_64_boxcar = np.loadtxt(file97_64_boxcar, skiprows=30, usecols=(0,1,2))
t97_64_boxcar = (data97_64_boxcar[:,0]*1e9)[0:-1-290] 
vel97_64_boxcar = data97_64_boxcar[:,1][0:-1-290]
vel97_unc_64_boxcar = data97_64_boxcar[:,2][0:-1-290] 
center_times97_64_boxcar, smoothed_velocity97_64_boxcar = smooth_conv(t97_64_boxcar, vel97_64_boxcar, 12)
center_times97_64_boxcar, smoothed_velocity97_64_boxcar = smooth_conv(center_times97_64_boxcar, smoothed_velocity97_64_boxcar, 1)
SG_smooth_data97_64_boxcar = savgol_filter(smoothed_velocity97_64_boxcar,80, 3)
smoothed_velocity97_64_boxcar = SG_smooth_data97_64_boxcar
#ax1.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


# 9.6 ns tau

file94_96_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10194/10194_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'
data94_96_boxcar = np.loadtxt(file94_96_boxcar, skiprows=30, usecols=(0,1))
t94_96_boxcar = (data94_96_boxcar[:,0]*1e9)[0:-1-290] 
vel94_96_boxcar = data94_96_boxcar[:,1][0:-1-290]
center_times94_96_boxcar, smoothed_velocity94_96_boxcar = smooth_conv(t94_96_boxcar, vel94_96_boxcar, 12)
center_times94_96_boxcar, smoothed_velocity94_96_boxcar = smooth_conv(center_times94_96_boxcar, smoothed_velocity94_96_boxcar, 1)
SG_smooth_data94_96_boxcar = savgol_filter(smoothed_velocity94_96_boxcar,80, 3)
smoothed_velocity94_96_boxcar = SG_smooth_data94_96_boxcar
#ax1.plot(center_times94_96[0:-1-55], smoothed_velocity94_96[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file95_96_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10195/10195_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'
data95_96_boxcar = np.loadtxt(file95_96_boxcar, skiprows=30, usecols=(0,1))
t95_96_boxcar = (data95_96_boxcar[:,0]*1e9)[0:-1-290] 
vel95_96_boxcar = data95_96_boxcar[:,1][0:-1-290]
center_times95_96_boxcar, smoothed_velocity95_96_boxcar = smooth_conv(t95_96_boxcar, vel95_96_boxcar, 12)
center_times95_96_boxcar, smoothed_velocity95_96_boxcar = smooth_conv(center_times95_96_boxcar, smoothed_velocity95_96_boxcar, 1)
SG_smooth_data95_96_boxcar = savgol_filter(smoothed_velocity95_96_boxcar,80, 3)
smoothed_velocity95_96_boxcar = SG_smooth_data95_96_boxcar 
#ax1.plot(center_times95_96[0:-1-55], smoothed_velocity95_96[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file96_96_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10196/10196_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'
data96_96_boxcar = np.loadtxt(file96_96_boxcar, skiprows=30, usecols=(0,1))
t96_96_boxcar = (data96_96_boxcar[:,0]*1e9)[0:-1-290] 
vel96_96_boxcar = data96_96_boxcar[:,1][0:-1-290]
center_times96_96_boxcar, smoothed_velocity96_96_boxcar = smooth_conv(t96_96_boxcar, vel96_96_boxcar, 12)
center_times96_96_boxcar, smoothed_velocity96_96_boxcar = smooth_conv(center_times96_96_boxcar, smoothed_velocity96_96_boxcar, 1)
SG_smooth_data96_96_boxcar = savgol_filter(smoothed_velocity96_96_boxcar,80, 3)
smoothed_velocity96_96_boxcar = SG_smooth_data96_96_boxcar 
#ax1.plot(center_times96_96[0:-1-55], smoothed_velocity96_96[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

file97_96_boxcar = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/10197_100kA_60ns_noremovesin_1000_1e6_tau9p6ns_alpha40ps_zeropad100x_-800_-400_ref_-400_start_gauss_boxcar_hist_ROI_30_width.csv'
data97_96_boxcar = np.loadtxt(file97_96_boxcar, skiprows=30, usecols=(0,1,2))
t97_96_boxcar = (data97_96_boxcar[:,0]*1e9)[0:-1-290] 
vel97_96_boxcar = data97_96_boxcar[:,1][0:-1-290]
vel97_unc_96_boxcar = data97_96_boxcar[:,2][0:-1-290] 
center_times97_96_boxcar, smoothed_velocity97_96_boxcar = smooth_conv(t97_96_boxcar, vel97_96_boxcar, 12)
center_times97_96_boxcar, smoothed_velocity97_96_boxcar = smooth_conv(center_times97_96_boxcar, smoothed_velocity97_96_boxcar, 1)
SG_smooth_data97_96_boxcar = savgol_filter(smoothed_velocity97_96_boxcar,80, 3)
smoothed_velocity97_96_boxcar = SG_smooth_data97_96_boxcar
#ax1.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

t_SG_smooth_data97_96_boxcar, deriv_97_96_boxcar, deriv2_97_96_boxcar = calc_SG_deriv(center_times97_96_boxcar, smoothed_velocity97_96_boxcar, 80, 3)






if 'ED1 tau window comparison' in mode:
    
    # hann
    ax1.plot(center_times94_48[0:-1-55], smoothed_velocity94_48[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='blue', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_48[0:-1-55], smoothed_velocity95_48[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='blue', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_48[0:-1-55], smoothed_velocity96_48[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='blue', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_48[0:-1-55], smoothed_velocity97_48[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='blue', markeredgecolor='black')  #color='#FFC400'
    
    ax1.plot(center_times94_96[0:-1-55], smoothed_velocity94_96[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_96[0:-1-55], smoothed_velocity95_96[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_96[0:-1-55], smoothed_velocity96_96[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='cyan', markeredgecolor='black')  #color='#FFC400'

    ax1.plot(center_times94_64[0:-1-55], smoothed_velocity94_64[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_64[0:-1-55], smoothed_velocity95_64[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_64[0:-1-55], smoothed_velocity96_64[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


    #boxcar
    ax2.plot(center_times94_48_boxcar[0:-1-55], smoothed_velocity94_48_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='blue', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times95_48_boxcar[0:-1-55], smoothed_velocity95_48_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='blue', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times96_48_boxcar[0:-1-55], smoothed_velocity96_48_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='blue', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_48_boxcar[0:-1-55], smoothed_velocity97_48_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='blue', markeredgecolor='black')  #color='#FFC400'

    ax2.plot(center_times94_96_boxcar[0:-1-55], smoothed_velocity94_96_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times95_96_boxcar[0:-1-55], smoothed_velocity95_96_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times96_96_boxcar[0:-1-55], smoothed_velocity96_96_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='cyan', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_96_boxcar[0:-1-55], smoothed_velocity97_96_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color='cyan', markeredgecolor='black')  #color='#FFC400'

    ax2.plot(center_times94_64_boxcar[0:-1-55], smoothed_velocity94_64_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times95_64_boxcar[0:-1-55], smoothed_velocity95_64_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times96_64_boxcar[0:-1-55], smoothed_velocity96_64_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_64_boxcar[0:-1-55], smoothed_velocity97_64_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'





if 'ED1 tiled' in mode:
    
    # hann
    ax1.plot(center_times94_48[0:-1-55], smoothed_velocity94_48[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_48[0:-1-55], smoothed_velocity95_48[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_48[0:-1-55], smoothed_velocity96_48[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_48[0:-1-55], smoothed_velocity97_48[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 4.8 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    
    ax2.plot(center_times94_64[0:-1-55], smoothed_velocity94_64[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,  color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times95_64[0:-1-55], smoothed_velocity95_64[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times96_64[0:-1-55], smoothed_velocity96_64[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,  color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    ax3.plot(center_times94_96[0:-1-55], smoothed_velocity94_96[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax3.plot(center_times95_96[0:-1-55], smoothed_velocity95_96[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax3.plot(center_times96_96[0:-1-55], smoothed_velocity96_96[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax3.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


    #boxcar
    ax4.plot(center_times94_48_boxcar[0:-1-55], smoothed_velocity94_48_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax4.plot(center_times95_48_boxcar[0:-1-55], smoothed_velocity95_48_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax4.plot(center_times96_48_boxcar[0:-1-55], smoothed_velocity96_48_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax4.plot(center_times97_48_boxcar[0:-1-55], smoothed_velocity97_48_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    ax5.plot(center_times94_64_boxcar[0:-1-55], smoothed_velocity94_64_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax5.plot(center_times95_64_boxcar[0:-1-55], smoothed_velocity95_64_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax5.plot(center_times96_64_boxcar[0:-1-55], smoothed_velocity96_64_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,  color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax5.plot(center_times97_64_boxcar[0:-1-55], smoothed_velocity97_64_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 6.4 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    ax6.plot(center_times94_96_boxcar[0:-1-55], smoothed_velocity94_96_boxcar[0:-1-55],'-',label=r'ED1 #10194 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax6.plot(center_times95_96_boxcar[0:-1-55], smoothed_velocity95_96_boxcar[0:-1-55],'-',label=r'ED1 #10195 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax6.plot(center_times96_96_boxcar[0:-1-55], smoothed_velocity96_96_boxcar[0:-1-55],'-',label=r'ED1 #10196 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax6.plot(center_times97_96_boxcar[0:-1-55], smoothed_velocity97_96_boxcar[0:-1-55],'-',label=r'ED1 #10197 Uncoated Al $\tau = 9.6 $ ns, $\alpha = 0.04$ ns, 80 pt SG filter, boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


if 'ED1 comp' in mode:


    ax1.plot(center_times94_96[0:-1-55], smoothed_velocity94_96[0:-1-55],'-',label=r'#10194–10197 uncoated Al, $\tau = 9.6 $ ns, Hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_96[0:-1-55], smoothed_velocity95_96[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_96[0:-1-55], smoothed_velocity96_96[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    ax1.plot(center_times94_96_boxcar[0:-1-55], smoothed_velocity94_96_boxcar[0:-1-55],'--',label=r'#10194–10197 uncoated Al, $\tau = 9.6 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_96_boxcar[0:-1-55], smoothed_velocity95_96_boxcar[0:-1-55],'--',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_96_boxcar[0:-1-55], smoothed_velocity96_96_boxcar[0:-1-55],'--',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_96_boxcar[0:-1-55], smoothed_velocity97_96_boxcar[0:-1-55],'--',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    

    ax2.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'--',label=r'#10197 uncoated Al, $\tau = 6.4 $ ns, Hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_96[0:-1-55], smoothed_velocity97_96[0:-1-55],'-',label=r'#10197 uncoated Al, $\tau = 9.6 $ ns, Hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    ax2.plot(center_times97_64_boxcar[0:-1-55], smoothed_velocity97_64_boxcar[0:-1-55],'--',label=r'#10197 uncoated Al, $\tau = 6.4 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color='blue', markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_96_boxcar[0:-1-55], smoothed_velocity97_96_boxcar[0:-1-55],'-',label=r'#10197 uncoated Al, $\tau = 9.6 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5,color='blue', markeredgecolor='black')  #color='#FFC400'


    ax2.fill_between(center_times97_96_boxcar, smoothed_velocity97_96_boxcar-vel97_unc_96_boxcar, smoothed_velocity97_96_boxcar+vel97_unc_96_boxcar, color='blue', alpha=0.1)
   
    ax2.fill_between(center_times97_64, smoothed_velocity97_64-vel97_unc_64, smoothed_velocity97_64+vel97_unc_64, color=(139/256, 10/256, 165/256), alpha=0.1)


    vel_sigma = 0.193 # 10197, boxcar, 9.6 ns tau, 12 pt mov avg, 80 pt SG
    sigma_3 = 3*vel_sigma
    sigma_5 = 5*vel_sigma
    sigma_10 = 10*vel_sigma

    #ax2.text(28,-0.65, r'$\sigma$', fontsize=12)
    ax2.text(60,-1.35, r'$3\sigma$', fontsize=12)
    #ax2.text(27.5,-2.2, r'$5\sigma$', fontsize=12)
    ax2.text(59.3,-2.8, r'$10\sigma$', fontsize=12)
    
    #ax2.axhline(-vel_sigma, color='grey',linestyle=':', lw=1 )
    ax2.axhline(-sigma_3, linestyle=':', color='grey', lw=1)
    #ax2.axhline(-sigma_5, linestyle=':', color='grey', lw=1)
    ax2.axhline(-sigma_10, linestyle=':', color='grey', lw=1)



if 'ED1 vel/accel/jerk' in mode:
    
    ax1.plot(center_times97_64[0:-1-55], smoothed_velocity97_64[0:-1-55],'-',label=r'Velocity ($\vec{v}$)',linewidth=1.5,color=(139/256, 10/256, 165/256))  #color='#FFC400'

    #ax1.plot(center_times97_64_boxcar[0:-1-55], smoothed_velocity97_64_boxcar[0:-1-55],'-',label=r'#10197 uncoated Al, $\tau = 6.4 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1.5, color='blue', markeredgecolor='black')  #color='#FFC400'
    
    t_deriv_97_64, deriv_97_64, deriv2_97_64 = calc_SG_deriv(center_times97_64, smoothed_velocity97_64, 80, 3)
    index_range = np.where((np.array(t_deriv_97_64) >= 55) & (np.array(t_deriv_97_64) <= 70))
    
    extrema_jerk_97 = np.array(deriv2_97_64)[index_range[0]]
    peaks_jerk, _ = find_peaks(extrema_jerk_97)
    valleys_jerk, _ = find_peaks(-extrema_jerk_97)
    extrema_jerk = np.sort(np.concatenate((peaks_jerk, valleys_jerk)))
    
    time_peaks = np.array(t_deriv_97_64)[index_range][extrema_jerk]
    jerk_peaks = np.array(deriv2_97_64)[index_range][extrema_jerk]
    
    twin2.axvline(time_peaks[1], color= 'gray', linewidth = 0.5)
    twin2.axvline(time_peaks[2], color= 'gray', linewidth = 0.5)
    
    twin2.axvspan(time_peaks[1], time_peaks[2], alpha=0.15, color='gray')
    
    
    twin1.plot(t_deriv_97_64, deriv_97_64, '-', label=r'Acceleration ($\vec{a}$)', linewidth=1.5, color='orange') 
    twin2.plot(t_deriv_97_64, deriv2_97_64, '-', label=r'Jerk ($d\vec{a}/dt$)', linewidth=1.5, color='blue')
    
    
    twin2.plot(time_peaks[1:-1], jerk_peaks[1:-1], 's', color='darkblue', markersize= 5)
    
    
    
    ax1.text(48.5,-105, r'#10197, $\tau = 6.4$ ns, Hann window', fontsize= 8)
    
    
    #--
    
    
    
    # 4.8 ns tau hann
    t_deriv_97_48, deriv_97_48, deriv2_97_48 = calc_SG_deriv(center_times97_48, smoothed_velocity97_48, 80, 3)
    ax2.plot(t_deriv_97_48, deriv_97_48, '--', label=r'$\vec{a}$ – $\tau = 4.8$ ns, Hann', linewidth=1.5, color='orange') 
    twin3.plot(t_deriv_97_48, deriv2_97_48, '--', label=r'$d\vec{a}/dt$ – $\tau = 4.8$ ns, Hann', linewidth=1.5, color='blue')
    
    # 6.4 ns tau hann
    ax2.plot(t_deriv_97_64, deriv_97_64, '-.', label=r'$\vec{a}$ – $\tau = 6.4$ ns, Hann', linewidth=1.5, color='orange') 
    twin3.plot(t_deriv_97_64, deriv2_97_64, '-.', label=r'$d\vec{a}/dt$ – $\tau = 6.4$ ns, Hann', linewidth=1.5, color='blue')
    
    # 4.8 ns tau boxcar
    t_deriv_97_48_boxcar, deriv_97_48_boxcar, deriv2_97_48_boxcar = calc_SG_deriv(center_times97_48_boxcar, smoothed_velocity97_48_boxcar, 80, 3)
    ax2.plot(t_deriv_97_48_boxcar, deriv_97_48_boxcar, '-', label=r'$\vec{a}$ – $\tau = 4.8$ ns, Boxcar', linewidth=1.5, color='orange') 
    twin3.plot(t_deriv_97_48_boxcar, deriv2_97_48_boxcar, '-', label=r'$d\vec{a}/dt$ – $\tau = 4.8$ ns, Boxcar', linewidth=1.5, color='blue')

    # 6.4 ns tau boxcar
    t_deriv_97_64_boxcar, deriv_97_64_boxcar, deriv2_97_64_boxcar = calc_SG_deriv(center_times97_64_boxcar, smoothed_velocity97_64_boxcar, 80, 3)
    ax2.plot(t_deriv_97_64_boxcar, deriv_97_64_boxcar, ':', label=r'$\vec{a}$ – $\tau = 6.4$ ns, Boxcar', linewidth=1.5, color='orange') 
    twin3.plot(t_deriv_97_64_boxcar, deriv2_97_64_boxcar, ':', label=r'$d\vec{a}/dt$ – $\tau = 6.4$ ns, Boxcar', linewidth=1.5, color='blue')

    ax2.text(48.5,-13, r'#10197', fontsize= 8)
     
    
if 'ED1 MHD' in mode:


    # ED1 all shots 9.6 ns Boxcar
    ax1.plot(center_times94_96_boxcar[0:-1-55], smoothed_velocity94_96_boxcar[0:-1-55],'-',label=r'#10194–10197 uncoated Al, $\tau = 9.6 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_96_boxcar[0:-1-55], smoothed_velocity95_96_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_96_boxcar[0:-1-55], smoothed_velocity96_96_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_96_boxcar[0:-1-55], smoothed_velocity97_96_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    
    # ED1 all shots 4.8 ns Boxcar
    ax2.plot(center_times94_48_boxcar[0:-1-55], smoothed_velocity94_48_boxcar[0:-1-55],'-',label=r'#10194–10197 uncoated Al, $\tau = 4.8 $ ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times95_48_boxcar[0:-1-55], smoothed_velocity95_48_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times96_48_boxcar[0:-1-55], smoothed_velocity96_48_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1, color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax2.plot(center_times97_48_boxcar[0:-1-55], smoothed_velocity97_48_boxcar[0:-1-55],'-',label=r'_nolegend_', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    

    # MHD data plotting
    ax1.plot(t94_mhd, v94_mhd,'--',lw=1.5, label= 'MHD Calculation', color= 'k')
    ax1.plot(t95_mhd, v95_mhd,'--', lw=1.5, color= 'k')
    ax1.plot(t96_mhd, v96_mhd,'--', lw=1.5, color= 'k')
    ax1.plot(t97_mhd, v97_mhd,'--', lw=1.5, color= 'k')
    
    ax2.plot(t94_mhd, v94_mhd,'--',lw=1.5, label= 'MHD Calculation', color= 'k')
    ax2.plot(t95_mhd, v95_mhd,'--', lw=1.5, color= 'k')
    ax2.plot(t96_mhd, v96_mhd,'--', lw=1.5, color= 'k')
    ax2.plot(t97_mhd, v97_mhd,'--', lw=1.5, color= 'k')


if 'MHD v/a/j' in mode:
    
    
    file_MHD_accel = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRE_dual/MHD_accel.csv'
    file_MHD_jerk = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRE_dual/MHD_jerk.csv'
    file_MHD_temp = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRE_dual/MHD_temp.csv'
    
    data_MHD_accel = np.loadtxt(file_MHD_accel, usecols=(0,1), delimiter=',')
    data_MHD_jerk = np.loadtxt(file_MHD_jerk, usecols=(0,1), delimiter=',')
    data_MHD_temp = np.loadtxt(file_MHD_temp, usecols=(0,1), delimiter=',')
    
    data_MHD_accel = np.array(data_MHD_accel)
    data_MHD_jerk = np.array(data_MHD_jerk)
    data_MHD_temp = np.array(data_MHD_temp)
    
    ax1.plot(t94_mhd, v94_mhd,'-',lw=2, label= 'MHD Velocity', color= 'k')
    
    twin1.plot(data_MHD_accel[:,0], data_MHD_accel[:,1], '-', linewidth = 1.5, label= 'MHD Acceleration', color= 'orange')
    twin2.plot(data_MHD_jerk[:,0], data_MHD_jerk[:,1], '-', linewidth = 1.5, label= 'MHD Jerk', color= 'blue')
    twin3.plot(data_MHD_temp[:,0], data_MHD_temp[:,1], '-', linewidth = 1.5, label= 'MHD Temperature', color= 'red')
    
    '''
    twin1.plot(data_MHD_accel[:,0][::20], data_MHD_accel[:,1][::20], '-', linewidth = 0.25,  color= 'orange')
    twin2.plot(data_MHD_jerk[:,0][::20], data_MHD_jerk[:,1][::20], '-', linewidth = 0.25, color= 'blue')
    twin3.plot(data_MHD_temp[:,0][::20], data_MHD_temp[:,1][::20], '-', linewidth = 0.25,  color= 'red')
    '''
    

if 'ED1 PDV MHD melt' in mode:
    
    PDV_shift_MHD = 3.6675

    # MHD 
    file_MHD_accel = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRE_dual/MHD_accel.csv'
    file_MHD_jerk = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/PRE_dual/MHD_jerk.csv'
    data_MHD_accel = np.loadtxt(file_MHD_accel, usecols=(0,1), delimiter=',')
    data_MHD_jerk = np.loadtxt(file_MHD_jerk, usecols=(0,1), delimiter=',')
    
    ax1.plot(t97_mhd, v97_mhd,'--',lw=1.5, label= 'MHD', color= 'k')
    #ax2.plot(data_MHD_accel[:,0], data_MHD_accel[:,1], '-', linewidth = 1.5, label= 'MHD Acceleration', color= 'orange')
    #twin2.plot(data_MHD_jerk[:,0], data_MHD_jerk[:,1], '-', linewidth = 1.5, label= 'MHD Jerk', color= 'blue')
    
    
    ax1.plot(center_times97_48_boxcar[0:-1-55] + PDV_shift_MHD, smoothed_velocity97_48_boxcar[0:-1-55],'-',label=r'Experiment', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    #ax1.plot(center_times97_64_boxcar[0:-1-55]+ PDV_shift_MHD, smoothed_velocity97_64_boxcar[0:-1-55],'-.',label=r'#10197, \tau = 6.4 ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'

    #ax1.plot(center_times97_48[0:-1-55] + PDV_shift_MHD, smoothed_velocity97_48[0:-1-55],'-',label=r'#10197, \tau = 4.8 ns, Hann', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(39/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    #ax1.plot(center_times97_64[0:-1-55] + PDV_shift_MHD, smoothed_velocity97_64[0:-1-55],'-.',label=r'#10197, \tau = 6.4 ns, Boxcar', markevery=50,markeredgewidth=0.75, markersize=4,markerfacecolor="none",linewidth=1,color=(39/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'


    # 4.8 ns tau hann
    t_deriv_97_48, deriv_97_48, deriv2_97_48 = calc_SG_deriv(center_times97_48, smoothed_velocity97_48, 80, 3)
    #ax2.plot(np.array(t_deriv_97_48)+ PDV_shift_MHD, deriv_97_48, '--', label=r'$\vec{a}$ – $\tau = 4.8$ ns, Hann', linewidth=1.5, color='orange') 
    #twin2.plot(np.array(t_deriv_97_48)+ PDV_shift_MHD, deriv2_97_48, '--', label=r'$d\vec{a}/dt$ – $\tau = 4.8$ ns, Hann', linewidth=1.5, color='blue')
    
    # 6.4 ns tau hann
    t_deriv_97_64, deriv_97_64, deriv2_97_64 = calc_SG_deriv(center_times97_64, smoothed_velocity97_64, 80, 3)
    #ax2.plot(np.array(t_deriv_97_64)+ PDV_shift_MHD, deriv_97_64, '-.', label=r'$\vec{a}$ – $\tau = 6.4$ ns, Hann', linewidth=1.5, color='orange') 
    #twin2.plot(np.array(t_deriv_97_64)+ PDV_shift_MHD, deriv2_97_64, '-.', label=r'$d\vec{a}/dt$ – $\tau = 6.4$ ns, Hann', linewidth=1.5, color='blue')
    
    # 4.8 ns tau boxcar
    t_deriv_97_48_boxcar, deriv_97_48_boxcar, deriv2_97_48_boxcar = calc_SG_deriv(center_times97_48_boxcar, smoothed_velocity97_48_boxcar, 80, 3)
    ax2.plot(np.array(t_deriv_97_48_boxcar)+ PDV_shift_MHD, deriv_97_48_boxcar, '-', label=r'$\vec{a}$ – $\tau = 4.8$ ns, Boxcar', linewidth=1.5, color='orange') 
    twin2.plot(np.array(t_deriv_97_48_boxcar)+ PDV_shift_MHD, deriv2_97_48_boxcar, '-', label=r'$d\vec{a}/dt$ – $\tau = 4.8$ ns, Boxcar', linewidth=1.5, color='blue')

    # 6.4 ns tau boxcar
    t_deriv_97_64_boxcar, deriv_97_64_boxcar, deriv2_97_64_boxcar = calc_SG_deriv(center_times97_64_boxcar, smoothed_velocity97_64_boxcar, 80, 3)
    #ax2.plot(np.array(t_deriv_97_64_boxcar)+ PDV_shift_MHD, deriv_97_64_boxcar, ':', label=r'$\vec{a}$ – $\tau = 6.4$ ns, Boxcar', linewidth=1.5, color='orange') 
    #twin2.plot(np.array(t_deriv_97_64_boxcar)+ PDV_shift_MHD, deriv2_97_64_boxcar, ':', label=r'$d\vec{a}/dt$ – $\tau = 6.4$ ns, Boxcar', linewidth=1.5, color='blue')

    

    twin2.text(61.5, -6, 'Solid', fontsize=11,bbox=dict(facecolor='white',alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    twin2.text(64.8, -6, 'Solid/Liquid', fontsize=11,bbox=dict(facecolor='white', alpha=1,edgecolor='black', boxstyle='square', lw=0.5))
    twin2.text(69.5, -6, 'Liquid', fontsize=11, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))

    twin2.text(62.4, 10, r'$\frac {d \vec{a}}{dt} > 0$', fontsize=14)
    twin2.text(68.3, 10, r'$\frac {d \vec{a}}{dt} < 0$', fontsize=14)

    
    ax2.axvline(64.4, lw = 0.75, color = 'gray')
    ax2.axvline(68.1, lw = 0.75, color = 'gray')
    twin2.axvspan(64.4, 68.1, alpha=0.15, color='gray')
    
    ax1.axvline(64.4, lw = 0.75, color = 'gray')
    ax1.axvline(68.1, lw = 0.75, color = 'gray')
    ax1.axvspan(64.4, 68.1, alpha=0.15, color='gray')
    
    p0=patches.FancyArrowPatch(((64.4), 120), (68.1, 120),lw=1, arrowstyle='<->', mutation_scale=12)
    ax1.add_patch(p0)
    ax1.text(64.7, 130, 'Melt Duration', fontsize=10)
    
    ax1.text(59.8, 135, r'#10197', fontsize = 8)
    ax1.text(60, 120, r't = t + 3.6676 ns', fontsize = 8)
    ax1.text(68.8, 15, r'$\tau = 4.8$ ns, Boxcar', fontsize = 8)
   


if 'ED1 PDV amplitude' in mode:
    
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

    
    #twin1.plot(amp_t_store, np.array(amplitudes94)*1e3)
    #twin1.plot(amp_t_store, np.array(amplitudes95)*1e3)
    #twin1.plot(amp_t_store, np.array(amplitudes96)*1e3)
    #twin1.plot(amp_t_store, np.array(amplitudes97)*1e3)



    amp_line, = twin1.plot(amp_t_store, scaled_amp_avg,'-s',ms=6, label='Average PDV Signal Amplitude (#10194–10197)', markerfacecolor='none', color='gray',lw= 1.5)


    ax1.plot(center_times94_48_boxcar[0:-1-55], smoothed_velocity94_48_boxcar[0:-1-55],'-',label=r'Velocity (#10194–10197)', linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times95_48_boxcar[0:-1-55], smoothed_velocity95_48_boxcar[0:-1-55],'-',label=r'_nolabel_', linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times96_48_boxcar[0:-1-55], smoothed_velocity96_48_boxcar[0:-1-55],'-',label=r'_nolabel_', linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'
    ax1.plot(center_times97_48_boxcar[0:-1-55], smoothed_velocity97_48_boxcar[0:-1-55],'-',label=r'_nolabel_', linewidth=1.5,color=(139/256, 10/256, 165/256), markeredgecolor='black')  #color='#FFC400'







### ---- plotting options section ----


if 'ED1 tiled' in mode:
    
    #ax1.legend(loc=2, fontsize=8)
    #ax2.legend(loc=2, fontsize=8)
    
    ax1.text(55,170, r'Hann', fontsize=10, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    ax4.text(55,170, r'Boxcar', fontsize=10, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    
    ax1.text(60.2,210, r'$\tau = 4.8$ ns', fontsize=10)
    ax2.text(60.2,210, r'$\tau = 6.4$ ns', fontsize=10)
    ax3.text(60.2,210, r'$\tau = 9.6$ ns', fontsize=10)
    

    ax1.set_xlim(54, 71)
    ax1.set_ylim(-25,200)
    ax2.set_xlim(54, 71)
    ax2.set_ylim(-25,200)
    ax3.set_xlim(54, 71)
    ax3.set_ylim(-25,200)
    ax4.set_xlim(54, 71)
    ax4.set_ylim(-25,200)
    ax5.set_xlim(54, 71)
    ax5.set_ylim(-25,200)
    ax6.set_xlim(54, 71)
    ax6.set_ylim(-25,200)
    
    twin1.set_ylim(-25,200)
    twin2.set_ylim(-25,200)
    twin3.set_ylim(-25,200)
    twin4.set_ylim(-25,200)
    twin5.set_ylim(-25,200)
    twin6.set_ylim(-25,200)
    
    ax1.xaxis.set_visible(False)
    ax2.xaxis.set_visible(False)
    ax3.xaxis.set_visible(False)
    
    ax2.yaxis.set_visible(False)
    ax3.yaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax6.yaxis.set_visible(False)
    
    twin1.yaxis.set_visible(False)
    twin2.yaxis.set_visible(False)
    twin4.yaxis.set_visible(False)
    twin5.yaxis.set_visible(False)
    
    
    #ax1.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    #ax2.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    
    
    ax4.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    ax5.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    ax6.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    
    ax1.set_ylabel('Velocity [m/s]', fontsize = 12)
    ax4.set_ylabel('Velocity [m/s]', fontsize = 12)
    
    twin3.set_ylabel('Current [kA]', fontsize = 12)
    twin6.set_ylabel('Current [kA]', fontsize = 12)
   
    
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    
    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_ED1_all_tile.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')


if 'ED1 tau window comparison' in mode:
    
    #ax1.legend(loc=2, fontsize=8)
    #ax2.legend(loc=2, fontsize=8)
    
    ax1.text(55,170, r'Hann', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))
    ax2.text(55,170, r'Boxcar', fontsize=12, bbox=dict(facecolor='white', alpha=1, edgecolor='black', boxstyle='square', lw=0.5))

    ax1.set_xlim(54, 72)
    ax1.set_ylim(-25,200)
    ax2.set_xlim(54, 72)
    ax2.set_ylim(-25,200)
    
    twin1.set_ylim(-25,200)
    twin2.set_ylim(-25,200)
    
    ax1.xaxis.set_visible(False)
    
    twin1.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    ax1.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    
    twin2.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    ax2.tick_params(axis='both', which='major', pad=2,size=1, labelsize=14)
    
    
    ax2.set_xlabel('Time ($t$) [ns]', fontsize = 15)
    
    ax1.set_ylabel('Velocity [m/s]', fontsize = 15)
    ax2.set_ylabel('Velocity [m/s]', fontsize = 15)
    
    twin1.set_ylabel('Current [kA]', fontsize = 15)
    twin2.set_ylabel('Current [kA]', fontsize = 15)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    
    fig.set_dpi(300)
    
    
if 'ED1 comp' in mode:
    
    ax1.xaxis.set_visible(False)
    
    ax1.legend(fontsize=7)
    ax2.legend(fontsize=7)
    
    ax1.set_xlim(25, 65)
    ax1.set_ylim(-10,50)
    ax2.set_xlim(25, 65)
    ax2.set_ylim(-6,6)

    twin1.set_ylim(-30,150)
    twin2.set_ylim(-100,100)
    
    ax1.yaxis.set_label_coords(-0.08, 0.5)
    ax2.yaxis.set_label_coords(-0.08, 0.5)


    twin1.yaxis.set_label_coords(1.12, 0.5)
    twin2.yaxis.set_label_coords(1.12, 0.5)

    ax2.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    
    
    ax1.set_ylabel('Velocity [m/s]', fontsize = 12)
    ax2.set_ylabel('Velocity [m/s]', fontsize = 12)
    
    twin1.set_ylabel('Current [kA]', fontsize = 12)
    twin2.set_ylabel('Current [kA]', fontsize = 12)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    
    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_ED1_compression.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')
    

if 'ED1 vel/accel/jerk' in mode:
    
    ax1.set_xlim(48,72)
    ax1.set_ylim(-125,250)
    
    twin1.set_ylim(-12.5,25)
    twin2.set_ylim(-5,10)
    
    twin1.yaxis.set_label_coords(1.12, 0.5)
    twin2.yaxis.set_label_coords(1.29, 0.5)
     
    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    twin1.tick_params(axis='y', which='major', labelsize=12)
    twin2.tick_params(axis='y', which='major', labelsize=12)
    twin3.tick_params(axis='y', which='major', labelsize=12)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = twin1.get_legend_handles_labels()
    handles3, labels3 = twin2.get_legend_handles_labels()
    
    all_handles = handles1 + handles2 + handles3
    all_labels = labels1 + labels2 + labels3
    
    ax1.legend(all_handles, all_labels, loc=2, fontsize = 8)
    
    

    handles1, labels1 = ax2.get_legend_handles_labels()
    handles2, labels2 = twin3.get_legend_handles_labels()

    all_handles = handles1 + handles2 
    all_labels = labels1 + labels2 
    
    ax2.legend(all_handles, all_labels, loc=2, fontsize = 7.2, ncol=1)
    

    twin2.spines['right'].set_position(('axes', 1.2))
    
    ax2.set_xlabel('Time ($t$) [ns]', fontsize = 14)
    ax1.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 14)
    twin1.set_ylabel(r'Acceleration ($\vec{a}$) [nm/ns$^2$]', fontsize=14)
    twin2.set_ylabel(r'Jerk ($d\vec{a} / dt$) [nm/ns$^3$]', fontsize=14)
    
    ax2.set_ylabel(r'Acceleration ($\vec{a}$) [nm/ns$^2$]', fontsize=14)
    twin3.set_ylabel(r'Jerk ($d\vec{a} / dt$) [nm/ns$^3$]', fontsize=14)
    
    ax2.set_xlim(48,72)
    ax2.set_ylim(-15,30)
    twin3.set_ylim(-6,12)

    #fig.tight_layout()
    
    plt.subplots_adjust(wspace=0.1, hspace=0.12)
     
    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_ED1_vel_accel_jerk.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')
    
    
if 'ED1 MHD' in mode:

    
    ax1.set_xlim(38, 67)
    ax1.set_ylim(-10,50)
    twin1.set_ylim(-40,200)
    
    ax2.set_xlim(53, 72)
    ax2.set_ylim(-25,200)
    twin2.set_ylim(-37.5,300)
    
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)
    
    
    ax2.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    
    ax1.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 12)
    ax2.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 12)
    
    twin1.set_ylabel('Current [kA]', fontsize = 12)
    twin2.set_ylabel('Current [kA]', fontsize = 12)
    
    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    twin1.tick_params(axis='y', which='major', labelsize=12)
    twin2.tick_params(axis='y', which='major', labelsize=12)
    
    ax1.legend(fontsize= 8, loc=2)
    ax2.legend(fontsize= 8, loc=2)
    
    
    plt.subplots_adjust(wspace=0.1, hspace=0.12)
    
    
    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/PDV_ED1_MHD.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')
    
    
    
if 'MHD v/a/j' in mode:
    
    
    ax1.set_xlim(62, 72)
    ax1.set_ylim(-100, 150)
    
    twin1.set_ylim(-20,30)
    twin2.set_ylim(-20,30)
    twin3.set_ylim(700,1200)
    
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    twin1.yaxis.set_label_coords(1.1, 0.5)
    twin2.yaxis.set_label_coords(1.3, 0.5)
    twin3.yaxis.set_label_coords(1.55, 0.5)
    
    twin2.spines['right'].set_position(('axes', 1.2))
    twin3.spines['right'].set_position(('axes', 1.4))
     
    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    twin1.tick_params(axis='y', which='major', labelsize=12)
    twin2.tick_params(axis='y', which='major', labelsize=12)
    twin3.tick_params(axis='y', which='major', labelsize=12)
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = twin1.get_legend_handles_labels()
    handles3, labels3 = twin2.get_legend_handles_labels()
    handles4, labels4 = twin3.get_legend_handles_labels()
    
    all_handles = handles1 + handles2 + handles3 + handles4
    all_labels = labels1 + labels2 + labels3 + labels4
    
    ax1.legend(all_handles, all_labels, loc=2, fontsize = 8.6)
    
    ax1.set_xlabel('Time ($t$) [ns]', fontsize = 14)
    ax1.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 14)
    twin1.set_ylabel(r'Acceleration ($\vec{a}$) [nm/ns$^2$]', fontsize=14)
    twin2.set_ylabel(r'Jerk ($d\vec{a} / dt$) [nm/ns$^3$]', fontsize=14)
    twin3.set_ylabel(r'Temperature ($T$) [K]', fontsize=14)
    
    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/ED1_MHD_vel_accel_jerk_temp.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')
    

if 'ED1 PDV MHD melt' in mode:
    
    
    ax1.set_xlim(59.5, 72.5)
    ax2.set_xlim(59.5, 72.5)
    
    ax1.set_ylim(-25, 150)
    
    twin1.set_ylim(-50, 300)
    
    
    ax2.set_ylim(-17.5,30)
    twin2.set_ylim(-7,12)


    ax1.tick_params(axis='y', which='major', labelsize=12)
    ax1.tick_params(axis='x', which='major', labelsize=12)
    ax2.tick_params(axis='y', which='major', labelsize=12)
    ax2.tick_params(axis='x', which='major', labelsize=12)
    twin1.tick_params(axis='y', which='major', labelsize=12)
    twin2.tick_params(axis='y', which='major', labelsize=12)
    
    ax2.set_xlabel('Time ($t$) [ns]', fontsize = 12)
    ax1.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 12)
    
    twin1.set_ylabel('Current [kA]', fontsize = 12)
    
    ax2.set_ylabel(r'Acceleration ($\vec{a}$) [nm/ns$^2$]', fontsize=12, color = 'darkorange')
    twin2.set_ylabel(r'Jerk ($d\vec{a} / dt$) [nm/ns$^3$]', fontsize=12, color = 'blue')
    
    plt.subplots_adjust(wspace=0.1, hspace=0.12)
    
    ax1.legend(loc=(0.2, 1.0), fontsize = 10, ncol = 2, frameon=False)
    
   
    ax2.spines['left'].set_color('darkorange')
    ax2.tick_params(axis='y', colors='darkorange')
    
    twin2.spines['left'].set_color('darkorange')
    twin2.spines['right'].set_color('blue')
    twin2.tick_params(axis='y', colors='blue')


    fig.set_dpi(300)

    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/ED1_PDV_MHD_melt.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')


if 'ED1 PDV amplitude' in mode:
    
    
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = twin1.get_legend_handles_labels()
    handles3, labels3 = twin2.get_legend_handles_labels()
    
    all_handles = handles1 + handles2 + handles3 
    all_labels = labels1 + labels2 + labels3 
    
    ax1.legend(all_handles, all_labels, loc=2, fontsize = 9)
    
    
    
    ax1.set_xlim(47, 72.5)
    
    ax1.set_ylim(-10, 250)
    twin1.set_ylim(100, 220)
    twin2.set_ylim(-10, 250)
    
    
    ax1.tick_params(axis='y', which='major', labelsize=14)
    ax1.tick_params(axis='x', which='major', labelsize=14)
    twin1.tick_params(axis='y', which='major', labelsize=14)
    twin2.tick_params(axis='y', which='major', labelsize=14)
    
    ax1.set_xlabel('Time ($t$) [ns]', fontsize = 14)
    ax1.set_ylabel(r'Velocity ($\vec{v}$) [m/s]', fontsize = 14)
    
    twin1.set_ylabel('Signal amplitude [mV]', fontsize = 14)
    twin2.set_ylabel('Current [kA]', fontsize = 14)


    twin2.spines['right'].set_position(('axes', 1.18))

    fig.set_dpi(300)
    
    #fig.savefig('/Users/Aidanklemmer/Desktop/HAWK/ED1_PDV_amplitude.pdf',format='pdf', facecolor='none',dpi=1000, bbox_inches='tight')



    


