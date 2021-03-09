function [trrs] =  TRRS_MRC(file,sf)

% clc; clear all; close all;
% Code to weight different wireless links based on the noise and signal
% TRRS and obtain a combined result.
% foldername = 'D:\WiCode\10172020_50samples_3segments\';
% filename = 'Yshape_loc2_NLOS_2_1.mat_phase_comp.mat';
% file = strcat(foldername,filename);
sf = 3;
debug = 1;
data  = load(file);
CSI = data.CSI_mtx(:,:,:,1:sf:end);
[trrs,trrs_ind] = calculate_TRRS(CSI,CSI,0,0.95);
trrs_ind_bak = trrs_ind;
% figure; imagesc(trrs);colormap jet; colorbar; caxis([0.95 1]);
outliers_index = [];
for i = 1:size(trrs_ind,2)
    current_link = trrs_ind{i};
    % Remove outliers
    check_vector = current_link(ceil(size(current_link,1)/2),1:end);
    [~,outliers] = hampel(1:length(check_vector),check_vector,5);
    current_link_trrs = abs(current_link);
    for j = 1:size(current_link_trrs,1)-1
        if (current_link_trrs(j,j+1)<0.95)
            outliers = [outliers ;j];
        end
    end
    outliers_index = [outliers_index ;outliers];
end

outliers_index(outliers_index==0) = [];

for i = 1:size(trrs_ind,2)
    current_link = abs(trrs_ind{i});
    current_link(outliers_index,:) = [];
    current_link(:,outliers_index) = [];
    trrs_ind{i} = current_link;
%     subplot(2,2,i);
%     imagesc(current_link);colormap jet; colorbar; caxis([0.95 1]);
end

% figure;
%% Estimate max signal amplitude in each link
signal_amp = zeros(1,size(trrs_ind,2));
for i = 1:size(trrs_ind,2)
%     subplot(2,2,i);
    current_link = trrs_ind{i};
    signal_amp(i) = 1-min(current_link(:));
    normalized_bessel = (current_link - min(current_link(:)))./(1-min(current_link(:)));
    trrs_ind{i} = normalized_bessel;
%     imagesc(normalized_bessel);colormap jet; colorbar; caxis([0 1]);
end


% Estimate the noise level in each link
matfilename = strrep(filename,'.mat_phase_comp.mat','.mat');
matfilename = strcat(foldername, matfilename);
motion_statistics = MotionStatistics5300_updated(matfilename,0,30,sf,0,outliers_index);

noise_estimation_indices = [];
for i = 1:length(motion_statistics)-50
    if (motion_statistics(i:i+50)<0)
        noise_estimation_indices = [noise_estimation_indices i];
    end
end

noise_amp = zeros(1,size(trrs_ind,2));
for i = 1:size(trrs_ind,2)
    current_link = trrs_ind{i};
    noise_estimation_matrix = current_link(noise_estimation_indices, noise_estimation_indices);
    noise_estimation_matrix = noise_estimation_matrix(:);
    noise_estimation_matrix(noise_estimation_matrix==1) = [];
    noise_amp(i) = 1-mean(noise_estimation_matrix);
end

% disp(['Signal amplitude: ',num2str(signal_amp)]);
% disp(['Noise amplitude: ',num2str(noise_amp)]);


% weight_factor = (signal_amp.^2)./(noise_amp.^2);
weight_factor = 1./(noise_amp.^2);
normalized_weights = weight_factor./sum(weight_factor);


for i = 1:size(trrs_ind,2)
    if (i==1)
        trrs_MRC = trrs_ind{i}*normalized_weights(i);
        original_trrs_link = abs(trrs_ind_bak{i});
        original_trrs_link(outliers_index,:) = [];
        original_trrs_link(:,outliers_index) = [];
        trrs_EGC = original_trrs_link*1/length(weight_factor);
    else
        trrs_MRC = trrs_MRC + trrs_ind{i}*normalized_weights(i);
        original_trrs_link = abs(trrs_ind_bak{i});
        original_trrs_link(outliers_index,:) = [];
        original_trrs_link(:,outliers_index) = [];
        trrs_EGC = trrs_EGC +  original_trrs_link*1/length(weight_factor);
    end
end

% Estimate the noise level in the trrs_MRC and trrs_EGC
noise_estimation_matrix = trrs_MRC(noise_estimation_indices, noise_estimation_indices);
noise_estimation_matrix = noise_estimation_matrix(:);
noise_estimation_matrix(noise_estimation_matrix==1) = [];
noise_amp_MRC = 1-mean(noise_estimation_matrix);

noise_estimation_matrix = trrs_EGC(noise_estimation_indices, noise_estimation_indices);
noise_estimation_matrix = noise_estimation_matrix(:);
noise_estimation_matrix(noise_estimation_matrix==1) = [];
noise_amp_EGC = 1-mean(noise_estimation_matrix);

% Estimate the signal level in the trrs_MRC and trrs_EGC

% current_matrix = trrs_MRC(:);
% signal_amp_MRC = 1-min(current_matrix);
% current_matrix = trrs_EGC(:);
% signal_amp_EGC = 1-min(current_matrix);

% figure;
% subplot(1,2,1);
% imagesc(trrs_MRC);colorbar; colormap jet; caxis([0.8 1]);
% title(['MRC: SNR:',num2str((signal_amp_MRC/noise_amp_MRC).^2)]);
% ax = gca; ax.FontSize = 12;
% 
% subplot(1,2,2);
% imagesc(trrs_EGC);colorbar; colormap jet; caxis([0.95 1]);
% title(['EGC: SNR:',num2str((signal_amp_EGC/noise_amp_EGC).^2)]);
% ax = gca; ax.FontSize = 12;



