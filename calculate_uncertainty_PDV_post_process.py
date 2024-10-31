"""
@author: Aidan klemmer
University of Nevada, Reno
aidanklemmer@outlook.com
10/29/24
"""



import pandas as pd
import numpy as np

# Define parameters
#shots = [10197]
shots = [10194, 10195, 10196, 10197]
taus = [4.8, 6.4, 9.6]
#taus = [3.2, 4.8, 6.4, 9.6]
#window_functions = ['hann']
window_functions = ['boxcar', 'hann']

N = len(shots) * len(taus) * len(window_functions)  # Total combinations

# Define the list of variables to process along with their corresponding uncertainty labels
variables = [
    {'name': 'val_max_accel_MC_mean', 'uncertainty': 'val_max_accel_MC_unc'},
    {'name': 'time_max_jerk_MC_mean', 'uncertainty': 'time_max_jerk_MC_unc'},
    {'name': 'time_min_jerk_MC_mean', 'uncertainty': 'time_min_jerk_MC_unc'},
    {'name': 'surf_B_start_melt_MC_mean', 'uncertainty': 'surf_B_start_melt_MC_unc'},
    {'name': 'surf_B_end_melt_MC_mean', 'uncertainty': 'surf_B_end_melt_MC_unc'}
]

# Define the file path
file_template = '/Users/Aidanklemmer/Desktop/HAWK/PDV_Paper_1/Excel_calc_dump/ED1_PDV_post_process_Monte_Carlo_error/Shot{shot}_{tau}ns_tau_{window}_calculated_parameters_Monte_Carlo_error_use_only.xlsx'

# Initialize a list to store results for each variable and each combination
all_results = []

# Loop over each variable to calculate statistics
for var in variables:
    var_name = var['name']
    uncertainty_name = var['uncertainty']
    
    # Loop through each combination of shot, tau, and window function to load data
    for shot in shots:
        for tau in taus:
            for window in window_functions:
                file_path = file_template.format(shot=shot, tau=tau, window=window)
                try:
                    # Load the data from the Excel file
                    df = pd.read_excel(file_path)
                    
                    # Extract and parse the variable and uncertainty values
                    value_str = df.loc[df['Variable'] == var_name, 'Value'].values[0]
                    uncertainty_str = df.loc[df['Variable'] == uncertainty_name, 'Value'].values[0]
                    
                    # Convert string representations to actual lists
                    value_list = eval(value_str) if isinstance(value_str, str) else [value_str]
                    uncertainty_list = eval(uncertainty_str) if isinstance(uncertainty_str, str) else [uncertainty_str]
                    
                    # Store results for the current variable
                    for value, uncertainty in zip(value_list, uncertainty_list):
                        all_results.append({
                            'shot': shot,
                            'tau': tau,
                            'window': window,
                            'variable': var_name,
                            'mean_value': value,
                            'uncertainty': uncertainty
                        })
                    
                    # Calculate and store "duration of melt" if this is the time_min_jerk variable
                    if var_name == 'time_min_jerk_MC_mean':
                        # Extract time_min_jerk and time_max_jerk values with their uncertainties
                        time_min_jerk = eval(df.loc[df['Variable'] == 'time_min_jerk_MC_mean', 'Value'].values[0])
                        time_min_jerk_uncertainty = eval(df.loc[df['Variable'] == 'time_min_jerk_MC_unc', 'Value'].values[0])
                        
                        time_max_jerk = eval(df.loc[df['Variable'] == 'time_max_jerk_MC_mean', 'Value'].values[0])
                        time_max_jerk_uncertainty = eval(df.loc[df['Variable'] == 'time_max_jerk_MC_unc', 'Value'].values[0])
                        
                        # Calculate the duration of melt and its uncertainty
                        duration = time_min_jerk[-1] - time_max_jerk[-1]
                        duration_uncertainty = np.sqrt(time_min_jerk_uncertainty[-1]**2 + time_max_jerk_uncertainty[-1]**2)
                        
                        # Store the duration of melt result
                        all_results.append({
                            'shot': shot,
                            'tau': tau,
                            'window': window,
                            'variable': 'duration_of_melt',
                            'mean_value': duration,
                            'uncertainty': duration_uncertainty
                        })

                except (FileNotFoundError, KeyError, IndexError) as e:
                    print(f"Data not found for {var_name} - Shot {shot}, Tau {tau} ns, Window {window}: {e}")
                    continue

# Convert all results into a DataFrame for easier inspection
df_all_results = pd.DataFrame(all_results)

# Calculate intra-option, inter-option, and total uncertainties for each shot and variable
summary_results = []
for shot in shots:
    for var in df_all_results['variable'].unique():
        shot_data = df_all_results[(df_all_results['shot'] == shot) & (df_all_results['variable'] == var)]
        
        # Average value across all combinations for the shot and variable
        mean_value = shot_data['mean_value'].mean()
        
        # Intra-option uncertainty (mean of uncertainties)
        U_intra = np.sqrt(np.mean(shot_data['uncertainty']**2))
        
        # Inter-option uncertainty (standard deviation of mean values), only if more than one data point
        if len(shot_data) > 1:
            sigma_inter = np.sqrt(np.sum((shot_data['mean_value'] - mean_value)**2) / (len(shot_data) - 1))
        else:
            sigma_inter = 0  # Set to zero if there's only one data point
        
        # Total uncertainty
        U_total = np.sqrt(U_intra**2 + sigma_inter**2)
        
        # Store the summary results
        summary_results.append({
            'shot': shot,
            'variable': var,
            'mean_value': mean_value,
            'U_intra': U_intra,
            'sigma_inter': sigma_inter,
            'U_total': U_total
        })

# Convert summary results into a DataFrame and display
df_summary_results = pd.DataFrame(summary_results)
print("\nSummary of Mean Values and Uncertainties for Each Variable and Shot:")
print(df_summary_results)








