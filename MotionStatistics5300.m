function [stats,F,stats_interpolate] = MotionStatistics5300(InputCSI,debug)
%% Compute the motion statistics

%% Parameters
F = 150;  % sampling frequency
DownSample = 1;
% outlier_thr = 0.01;
lag_index = 5;
LengthofBlock = round(2*F/DownSample);  OverlapRatio = 0.5;   Delta = round((1-OverlapRatio)*LengthofBlock);
Ng = 1;
eta = 0.02;

%% Read data

CSI = CSI_Preprocess(InputCSI);

%% Motion statistics detection
dimension_CSI = size(CSI);
kk = 1;  ll = 1;
while(kk+LengthofBlock-1 <= dimension_CSI(2))
    CSI_Block = squeeze(CSI(1:round(dimension_CSI(1)/1),kk:kk+LengthofBlock-1));
    %% outlier removal
    CSI_Block_Previous = CSI_Block(:,1:end-lag_index) - repmat(mean(CSI_Block(:,1:end-lag_index),2),1,size(CSI_Block(:,1:end-lag_index),2));
    CSI_Block_After = CSI_Block(:,lag_index+1:end) - repmat(mean(CSI_Block(:,lag_index+1:end),2),1,size(CSI_Block(:,lag_index+1:end),2));
    Correlation_Numerator = sum(CSI_Block_Previous.*CSI_Block_After,2);
    Correlation_Denominator = sqrt(sum(CSI_Block_Previous.^2,2)).*sqrt(sum(CSI_Block_After.^2,2));
    if ll == 1
        Motion.Correlation_Coefficient = Correlation_Numerator./ (Correlation_Denominator + eps);
    else
        Motion.Correlation_Coefficient = [Motion.Correlation_Coefficient, Correlation_Numerator./ (Correlation_Denominator + eps)];
    end
    Motion.Vote(ll) = mean(Motion.Correlation_Coefficient(Motion.Correlation_Coefficient(:,ll)~=0,ll));
    kk = kk+Delta;
    ll = ll+1;
end


%% Plotting figures
if (debug)
    figure,
    plot((1:length(Motion.Vote))*Delta*DownSample/F, Motion.Vote, '--o','LineWidth', 2)
    grid on
    xlabel('Time (second)','FontSize', 20)
    ylabel('Average Motion Statistics', 'FontSize', 20)
    ylim([-0.2 1.2])
end

stats = Motion.Vote;

stats_interpolate = repmat(stats,F,1);
stats_interpolate = reshape(stats_interpolate,1,length(stats)*F);
stats_interpolate = [stats_interpolate repmat(stats_interpolate(end),1,size(InputCSI,4)-length(stats_interpolate))];