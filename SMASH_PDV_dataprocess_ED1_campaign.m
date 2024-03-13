%% PDV data processing with SMASH 
%Aidan Klemmer
%University of Nevada, Reno
%Aidanklemmer@outlook.com
%3/13/24

%SMASH Library- Daniel Dolan (Sandia National Laboratories)

close all
clearvars
import SMASH.*

% PDV data directory location
pdvDirectory = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV/';

%shot selection
% ED1 campaign shots: 10194, 10195, 10196, and 10197
shot_list = 10197;

fprintf(['Working on Shot# ' num2str(shot_list) '\n' ]);

%file handling
datawfm_1 = ['shot' num2str(shot_list) '_Ch1.wfm']; % channel #1 used
wfmloc_1 =[pdvDirectory datawfm_1];

%create PDV objects
S1 = SMASH.Velocimetry.PDV(wfmloc_1, 'tektronix', 1);
S1.Name = ['Shot_' num2str(shot_list) '_Channel 1'];   

%crop data
%crop temporally
S1=crop(S1,[-400e-9 400e-9]);

%calculate offsets
f1 = 0.2e9;
f2 = 2.5e9;
t1 = -400e-9;
t2 = -200e-9;
S1 =calculateOffset(S1,'Frequency',[f1 f2], 'Time', [t1 t2]);

%shift time (shot dependant time correction)
%this shift connects Mykonos LTD "machine time" to the PDV diagnostic time
shift_file = '/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/ED_1_PDV_shifts_100kA_60ns.csv';
shifts = csvread(shift_file, 0, 0);
k = find((shifts(:,1) == shot_list) == 1);
t_shft_Ch1 = shifts(k, 2);
S1 =shiftTime(S1, t_shft_Ch1*1e-9);

%select reference region
S1 =selectReference(S1,[-400e-9 -200e-9]);

%FFT and Spectrogram settings. Partitioning is described by six parameters, any two of which define the other four.
S1 =partition(S1,'Duration',[3.2e-9, 40e-12]); % the values here are organized as [tau, alpha], where tau is the STFT window length/duration, and alpha is the spacing 
%between adjacent STFT window segments. 40 ps corresponds to the digitizer
%rate (1/25 GHz) so its the not skipping over any windows. 

%define region of interest 
R = SMASH.ROI.Curve();
roi_width = 250;
roi_shift = 0;

if ~any(strcmp(shot_list, '10197'))
    ROI = define(R, [-1e-7 0+roi_shift roi_width; 0 0+roi_shift roi_width; 54e-9 -5+roi_shift roi_width; 57.2e-9 -0.5+roi_shift roi_width; 59.6e-9 5.2+roi_shift roi_width; 61.2e-9 19.8+roi_shift roi_width; 63.2e-9 59.5+roi_shift roi_width;...
        65.2e-9 97.1+roi_shift roi_width; 67.2e-9 128+roi_shift roi_width; 68e-9 140.2+roi_shift roi_width; 69.2e-9 159+roi_shift roi_width; 71.2e-9 197.7+roi_shift roi_width; 73.2e-9 241.2+roi_shift roi_width;...
        76.4e-9 315.6+roi_shift roi_width; 78.4e-9 373.6+roi_shift roi_width; 80.4e-9 444+roi_shift roi_width; 82.2e-9 529.4+roi_shift roi_width; 84.4e-9 629.8+roi_shift roi_width;...
        87.2e-9 775.2+roi_shift roi_width; 90.4e-9 1022+roi_shift roi_width; 94.4e-9 1476+roi_shift roi_width; 110e-9 1500+roi_shift roi_width]);
end
if ~any(strcmp(shot_list, '10196'))
    ROI = define(R, [-1e-7 0 roi_width; 0 0 roi_width; 54e-9 -5 roi_width; 57.2e-9 -3 roi_width; 61.2e-9 14 roi_width; 63.2e-9 49.6 roi_width;...
        65.2e-9 82.3 roi_width; 67.2e-9 113.6 roi_width; 68e-9 126.3 roi_width; 69.2e-9 144.2 roi_width; 71.2e-9 175.2 roi_width;...
        73.2e-9 211 roi_width; 76.4e-9 276 roi_width; 78.4e-9 328 roi_width; 80.4e-9 389 roi_width; 84.4e-9 560 roi_width;...
        87.2e-9 715 roi_width; 90.4e-9 899 roi_width; 94.4e-9 1326 roi_width; 110e-9 1400 roi_width]);
end
if ~any(strcmp(shot_list, '10195'))
    ROI = define(R, [-1e-7 0 roi_width; 0 0 roi_width; 54e-9 -3.5 roi_width; 57e-9 -2.85 roi_width; 61.2e-9 15.5 roi_width; 63.2e-9 40.9 roi_width; 65.2e-9 88.5 roi_width; 67.2e-9 128.4 roi_width; ...
        68e-9 136.8 roi_width; 69.2e-9 145 roi_width; 71.2e-9 165 roi_width; 73.2e-9 189 roi_width; 74.4e-9 212 roi_width; 76e-9 248 roi_width; 78.2e-9 400 roi_width; 80e-9 420 roi_width; ...
        84.4e-9 662 roi_width; 90.4e-9 1084 roi_width; 94.4e-9 1267 roi_width; 110e-9 1400 roi_width]);
end
if ~any(strcmp(shot_list, '10194'))
    ROI = define(R, [-1e-7 0 roi_width; 0 0 roi_width; 54e-9 -5 roi_width; 57e-9 0 roi_width; 61.2e-9 10 roi_width; 63.2e-9 48 roi_width; 65.2e-9 93.5 roi_width; 67.2e-9 115 roi_width; ...
        68e-9 124 roi_width; 69.2e-9 134.9 roi_width; 71.2e-9 171 roi_width; 73.2e-9 213 roi_width; 76.2e-9 295 roi_width; 78.2e-9 350 roi_width; 80.2e-9 419.3 roi_width; ...
        84.2e-9 665 roi_width; 90.2e-9 1081 roi_width; 94.2e-9 1328 roi_width; 110e-9 1400 roi_width]);
end
%apply ROI
S1 =selectROI(S1, ROI); 

%generate velocity history
S1 =generateHistory(S1);

%write raw history to csv
loc='/Users/Aidanklemmer/Desktop/HAWK/UNR_Research/UNR_Group/Engineered_Defects/ED1/PDV_Results/Shot_10197/';
export(S1, [loc num2str(shot_list) '_100kA_60ns_noremovesin_1000_1e6_tau3_2ns_alpha40ps_zeropad100x_-400_-200_gauss_boxcar_hist_ROI_250_width.csv']);

