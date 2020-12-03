function [motion_statistics] = MotionStatistics5300_updated(filename,phase_boosting,num_useful_subcarriers,sf,debug,outliers)
% Function to calculate the correlation matrix given a CSI time series and
% reference CSI time series.
data = load(filename);
CSI = data.csi_trace.csi(:,:,:,1:sf:end);
CSI(:,:,:,outliers) = [];
mactimer = data.csi_trace.mactimer(1:sf:end);
mactimer(outliers) = [];
channel = CSI_phase_boosting(CSI,mactimer, 1, 1,5.8e9,phase_boosting);    % CSI preprocessing (normalization, downsample...)
csi_power_matrix = channel.csi_power_matrix_;
N = size(csi_power_matrix,2);
nstep = 1;
nwindow = 50;
motion_statistics = zeros(1,length(1:nstep:N));
for i = 1:nstep:N
    G = csi_power_matrix(:,max(i-nwindow+1,2):i);
    
    CSI_Block = G;
    CSI_Block = CSI_Block - repmat(mean(CSI_Block,2),1,size(CSI_Block,2));
    for lag = 0 : 1
        NumofSampleforCorr = size(CSI_Block, 2) - lag;
        subcarrier_sensitivity_matrix( : , lag+1) = mean(CSI_Block( : , end-NumofSampleforCorr-lag+1 : end-lag) .* CSI_Block(:, end-NumofSampleforCorr+1 : end), 2);
    end
        subcarrier_sensitivity_tmp = subcarrier_sensitivity_matrix(:, 2) ./ (subcarrier_sensitivity_matrix(:, 1) + eps);

            
    sorted_SS = sort(subcarrier_sensitivity_tmp,'descend');
%     weight_selected = sorted_SS(1:num_useful_subcarriers);
%     
%     
%     %% MRC combining
%     acf_mrc_weight = max(weight_selected, 0) / (norm(max(weight_selected, 0), 1) + eps);    % The normalized MRC weights
%     ms_weighted = acf_mrc_weight.*weight_selected;    % MRC combining step 1
    motion_statistics(i) = mean(sorted_SS);
end
figure;
plot((mactimer-mactimer(1))/1e6, motion_statistics,'LineWidth',2,'LineStyle','--'); hold on;grid on;
plot((mactimer-mactimer(1))/1e6, smooth(motion_statistics,0.2,'rloess'),'k-','LineWidth',1); hold on;grid on;
xlabel('Time(s)');ylabel('Motion Statistics');
ylim([-0.05 0.3]);
xlim([0,(mactimer(end)-mactimer(1))/1e6]);
ax = gca; ax.FontSize = 14;
legend('Raw','Smoothed');
end

