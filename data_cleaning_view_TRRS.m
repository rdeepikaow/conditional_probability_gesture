%% Author: Sai Deepika Regani
% Date: October 29th, 2020
% Main script for gesture classification using conditional probability
% approach.
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc; clear all; close all;
addpath('C:\Users\origingpu\Desktop\5GHz_Gesture');
debug = 0;
params_gesture_construction.sf = 4; % Sampling factor
params_gesture_construction.display_min = 0.95; % caxis min for display
params_gesture_construction.outlier_removal_offset = 1; % lag in samples outlier removal
params_gesture_construction.outlier_removal_th = 0.92; % Similarity th for outlier removal
params_gesture_construction.debug = 0;
params_gesture_construction.crop_index = -1; % Index at which CSI time series is cropped
params_gesture_construction.num_useful_subcarriers = 248; % Applicable for MRC and correlation
params_gesture_construction.similarity_method = 'trrs'; %'trrs','correlation_phaseboost','correlation'


if (strcmp(params_gesture_construction.similarity_method,'trrs'))
    params_gesture_segmentation.ccf_similarity_decay_th = 0.985;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.002;
else
    params_gesture_segmentation.ccf_similarity_decay_th = 0.6;
    params_gesture_segmentation.ccf_similarity_decay_step = 0.1;
end

params_gesture_segmentation.ccf_similarity_decay_smooth_win = 100;
params_gesture_segmentation.sounding_rate_set = 1500; %Hz
params_angle_estimation.sounding_rate_set = params_gesture_segmentation.sounding_rate_set;
params_gesture_segmentation.min_segment_duration = 0.3; %sec
params_gesture_segmentation.local_peak_win = 96;
params_gesture_segmentation.debug = 0;

direc = dir(fullfile('D:\WiCode\10192020_additional_data_to_10172020','*Pshape*NLOS*.mat_phase_comp.mat'));


for ff = 1:size(direc,1)
    filename = strcat(direc(ff).folder,'/',direc(ff).name);
    matfilename = filename;
    if contains(filename,'phase_comp.mat')
        matfilename = strrep(filename,'_phase_comp.mat','');
    end
    extractedGT = extractBetween(filename,'/','shape');
    disp(['Processing: ', direc(ff).name]);
    params_gesture_construction.matfilename = matfilename;
    params_gesture_construction.filename = filename;
    gesture_construction = GestureConstruction(params_gesture_construction);
    gesture_construction = gesture_construction.preprocess(filename);
    figure; imagesc(gesture_construction.similarity_matrix_); colorbar; caxis([0.95 1]);
    
    user_input = input('Is the data faulty?: ');
    if ~isempty(user_input)
        movedirectory = strcat(direc(ff).folder,'/faulty_02_25');
        if exist(movedirectory,'dir')~=7
            mkdir(movedirectory);
        end
        movefilename = strcat(movedirectory,'/',direc(ff).name);
        movefile(filename, movefilename);
        
        matfilename = strrep(filename,'_phase_comp.mat','');
        move_matfilename = strrep(movefilename,'_phase_comp.mat','');
        movefile(matfilename,move_matfilename);
    end
    close all;
end
