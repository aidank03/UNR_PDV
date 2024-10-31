#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:28:40 2024

@author: Aidanklemmer
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

# Define shot numbers, tau values, Monte Carlo iterations, colors, and window functions
shots = [10194, 10195, 10196, 10197]

taus = [9.6]
mc_iterations = [10000]
colors = ['green', 'blue', 'cyan', 'magenta', 'red']
window_functions = ['hann']

# Define the file path template (assuming a consistent naming convention)
file_template = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/Excel_calc_dump/ED1_PDV_post_process_Monte_Carlo_full_samples_TEST/Shot{shot}_{tau}ns_tau_{window}_calculated_parameters_Monte_Carlo_error_Full_Sample_TEST_10000_iter.csv'



# Loop through each shot and window function and create a 2x2 grid of plots for each combination
for shot in shots:
    for window in window_functions:
        
        
        '''
        # Create figures for the current shot and window function
        fig, axs = plt.subplots(2, 2, figsize=(8, 5))
        fig.suptitle(f'Error in max acceleration for ED1 shot {shot} - {window.capitalize()} window')
        fig.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig2, axs2 = plt.subplots(2, 2, figsize=(8, 5))
        fig2.suptitle(f'Error in time max jerk for ED1 shot {shot} - {window.capitalize()} window')
        fig2.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig3, axs3 = plt.subplots(2, 2, figsize=(8, 5))
        fig3.suptitle(f'Error in time min jerk for ED1 shot {shot} - {window.capitalize()} window')
        fig3.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig4, axs4 = plt.subplots(2, 2, figsize=(8, 5))
        fig4.suptitle(f'Error in surface B-field at time max jerk for ED1 shot {shot} - {window.capitalize()} window')
        fig4.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig5, axs5 = plt.subplots(2, 2, figsize=(8, 5))
        fig5.suptitle(f'Error in surface B-field at time min jerk for ED1 shot {shot} - {window.capitalize()} window')
        fig5.subplots_adjust(wspace=0.4, hspace=0.45)
        
        '''
        
        
        fig6, ax6 = plt.subplots(1, 1, figsize=(6, 4))
        fig6.suptitle(f'MC 10000 iterations max acceleration for ED1 shot {shot}')
        #fig6.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig7, ax7 = plt.subplots(1, 1, figsize=(6, 4))
        fig7.suptitle(f'EMC 10000 iterations time max jerk for ED1 shot {shot}')
        #fig7.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig8, ax8 = plt.subplots(1, 1, figsize=(6, 4))
        fig8.suptitle(f'MC 10000 iterations time min jerk for ED1 shot {shot}')
        #fig8.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig9, ax9 = plt.subplots(1, 1, figsize=(6, 4))
        fig9.suptitle(f'MC 10000 iterations surface B-field at time max jerk for ED1 shot {shot}')
        #fig9.subplots_adjust(wspace=0.4, hspace=0.45)
        
        fig10, ax10 = plt.subplots(1, 1, figsize=(6, 4))
        fig10.suptitle(f'MC 10000 iterations surface B-field at time min jerk for ED1 shot {shot}')
        #fig10.subplots_adjust(wspace=0.4, hspace=0.45)
        '''
        
        # Set DPI for high-quality output
        
        fig.set_dpi(300)
        fig2.set_dpi(300)
        fig3.set_dpi(300)
        fig4.set_dpi(300)
        fig5.set_dpi(300)
        '''
        fig6.set_dpi(300)
        fig7.set_dpi(300)
        fig8.set_dpi(300)
        fig9.set_dpi(300)
        fig10.set_dpi(300)

        
        
        # Loop through each tau value and plot on the corresponding subplot
        for i, tau in enumerate(taus):
            # Generate the file path for the current shot, tau, and window function
            file_path = file_template.format(shot=shot, tau=tau, window=window)
            
            # Load the data
            try:
                df = pd.read_csv(file_path)
                
                # Extract values for each error type
                errors1 = eval(df.loc[df['Variable'] == 'val_max_accel_MC_unc', 'Value'].values[0])
                errors2 = eval(df.loc[df['Variable'] == 'time_max_jerk_MC_unc', 'Value'].values[0])
                errors3 = eval(df.loc[df['Variable'] == 'time_min_jerk_MC_unc', 'Value'].values[0])
                errors4 = eval(df.loc[df['Variable'] == 'surf_B_start_melt_MC_unc', 'Value'].values[0])
                errors5 = eval(df.loc[df['Variable'] == 'surf_B_end_melt_MC_unc', 'Value'].values[0])
                
                error_all_values_6 = eval(df.loc[df['Variable'] == 'val_max_accel_MC_full_samp_vals', 'Value'].values[0])
                error_all_values_7 = eval(df.loc[df['Variable'] == 'time_max_jerk_MC_full_samp_vals', 'Value'].values[0])
                error_all_values_8 = eval(df.loc[df['Variable'] == 'time_min_jerk_MC_full_samp_vals', 'Value'].values[0])
                error_all_values_9 = eval(df.loc[df['Variable'] == 'surf_B_start_melt_MC_full_samp_vals', 'Value'].values[0])
                error_all_values_10 = eval(df.loc[df['Variable'] == 'surf_B_end_melt_MC_full_samp_vals', 'Value'].values[0])
                
            
            except (FileNotFoundError, KeyError, IndexError):
                print(f"Could not process file: {file_path}")
                continue
            
            markers = ['o', 's', '^', 'D']  # Define distinct markers for each tau value
            
            '''
            
            # Plot on the appropriate subplot for each figure with unique markers and black outlines
            ax = axs[i // 2, i % 2]
            ax.scatter(mc_iterations, errors1, color=colors, marker='D', s=50, edgecolors='black', linewidths=0.5, label=[f'{iter} MC iter' for iter in mc_iterations])
            ax.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax.set_xlabel('Monte Carlo iterations')
            ax.set_ylabel('Error [nm/ns²]')
            
            ax2 = axs2[i // 2, i % 2]
            ax2.scatter(mc_iterations, errors2, color=colors, marker='D', s=50, edgecolors='black', linewidths=0.5, label=[f'{iter} MC iter' for iter in mc_iterations])
            ax2.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax2.set_xlabel('Monte Carlo iterations')
            ax2.set_ylabel('Error [ns]')
            
            ax3 = axs3[i // 2, i % 2]
            ax3.scatter(mc_iterations, errors3, color=colors, marker='D', s=50, edgecolors='black', linewidths=0.5, label=[f'{iter} MC iter' for iter in mc_iterations])
            ax3.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax3.set_xlabel('Monte Carlo iterations')
            ax3.set_ylabel('Error [ns]')
            
            ax4 = axs4[i // 2, i % 2]
            ax4.scatter(mc_iterations, errors4, color=colors, marker='D', s=50, edgecolors='black', linewidths=0.5, label=[f'{iter} MC iter' for iter in mc_iterations])
            ax4.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax4.set_xlabel('Monte Carlo iterations')
            ax4.set_ylabel('Error [T]')
    
            ax5 = axs5[i // 2, i % 2]
            ax5.scatter(mc_iterations, errors5, color=colors, marker='D', s=50, edgecolors='black', linewidths=0.5, label=[f'{iter} MC iter' for iter in mc_iterations])
            ax5.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax5.set_xlabel('Monte Carlo iterations')
            ax5.set_ylabel('Error [T]')
            
            '''
            
            # histogram
            


            

            

            ax6.hist(error_all_values_6, bins=50, range = [np.min(error_all_values_6), np.max(error_all_values_6)], label=[f'{iter} MC iter' for iter in mc_iterations])
            ax6.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax6.set_xlabel('Max acceleration during melt [nm/ns$^2$]')
            ax6.set_ylabel('Counts')
            

            ax7.hist(error_all_values_7, bins=50, range = [np.min(error_all_values_7), np.max(error_all_values_7)], label=[f'{iter} MC iter' for iter in mc_iterations])
            ax7.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax7.set_xlabel('Time max jerk [ns]')
            ax7.set_ylabel('Counts')
            

            ax8.hist(error_all_values_8, bins=50, range = [np.min(error_all_values_8), np.max(error_all_values_8)], label=[f'{iter} MC iter' for iter in mc_iterations])
            ax8.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax8.set_xlabel('Time min jerk [ns]')
            ax8.set_ylabel('Counts')
            

            ax9.hist(error_all_values_9, bins=50, range = [np.min(error_all_values_9), np.max(error_all_values_9)], label=[f'{iter} MC iter' for iter in mc_iterations])
            ax9.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax9.set_xlabel('Surface B-field at time max jerk [T]')
            ax9.set_ylabel('Counts')
            ax9.set_xticks(np.arange(np.min(error_all_values_9), np.max(error_all_values_9), step=0.0005))
            ax9.set_xticklabels(np.arange(np.min(error_all_values_9), np.max(error_all_values_9), step=0.0005))
            ax9.set_xlim(np.min(error_all_values_9), np.max(error_all_values_9))
            ax9.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))

            ax10.hist(error_all_values_10, bins=50, range = [np.min(error_all_values_10), np.max(error_all_values_10)], label=[f'{iter} MC iter' for iter in mc_iterations])
            ax10.set_title(f'{window.capitalize()} τ = {tau} ns')
            ax10.set_xlabel('Surface B-field at time min jerk [T]')
            ax10.set_ylabel('Counts')
            ax10.set_xticks(np.arange(np.min(error_all_values_10), np.max(error_all_values_10), step=0.0005))
            ax10.set_xticklabels(np.arange(np.min(error_all_values_10), np.max(error_all_values_10), step=0.0005))
            ax10.set_xlim(np.min(error_all_values_10), np.max(error_all_values_10))
            ax10.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            
            

# Show or save each figure if needed
plt.show()  # Show all figures for the current shot and window function

    
    


    
    
    
    
    
    
