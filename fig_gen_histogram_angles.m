clc; clear all; close all;
data_zero = load('E:\deepika\WiCode\angle_classification_entire_dataset\zero_angle/output_sf_3.mat');
output_zero = data_zero.output;
datalength_zero = size(output_zero.gesture_data,2);

area_ratio_list_zero = [];
peaks_detected_length_list_zero = [];
for i = 1:datalength_zero
    if (~isfield(output_zero.gesture_data{i},'angle_data'))
        continue;
    end
    num_angles = length(output_zero.gesture_data{i}.angle_data);
    for a = 1:num_angles
        
        if (isempty(output_zero.gesture_data{i}.angle_data{a}))
            continue;
        end
        current_entry = output_zero.gesture_data{i}.angle_data{a}.area_ratio;
        area_ratio_list_zero = [area_ratio_list_zero current_entry];
        
        current_entry = output_zero.gesture_data{i}.angle_data{a}.peaks_length;
        peaks_detected_length_list_zero = [peaks_detected_length_list_zero current_entry];
    end
end

% area_ratio_list_zero = data_zero.feature.area_ratio_list;
% peaks_detected_length_list_zero = data_zero.feature.peaks_detected_length_list;
% peaks_detected_length_list_zero(area_ratio_list_zero==Inf) = [];
% area_ratio_list_zero(area_ratio_list_zero==Inf) = [];


data_acute = load('E:\deepika\WiCode\angle_classification_entire_dataset\acute_angle/output_sf_3.mat');
output_acute  = data_acute.output;
datalength_acute = size(output_acute.gesture_data,2);

area_ratio_list_acute = [];
peaks_detected_length_list_acute = [];
for i = 1:datalength_acute
    if (~isfield(output_acute.gesture_data{i},'angle_data'))
        continue;
    end
    num_angles = length(output_acute.gesture_data{i}.angle_data);
    for a = 1:num_angles
        if (isempty(output_acute.gesture_data{i}.angle_data{a}))
            continue;
        end
        current_entry = output_acute.gesture_data{i}.angle_data{a}.area_ratio;
        area_ratio_list_acute = [area_ratio_list_acute current_entry];
        
        current_entry = output_acute.gesture_data{i}.angle_data{a}.peaks_length;
        peaks_detected_length_list_acute = [peaks_detected_length_list_acute current_entry];
    end
end




% area_ratio_list_acute = data_acute.feature.area_ratio_list;
% peaks_detected_length_list_acute = data_acute.feature.peaks_detected_length_list;
% peaks_detected_length_list_acute(area_ratio_list_acute==Inf) = [];
% area_ratio_list_acute(area_ratio_list_acute==Inf) = [];


% figure;
% t1 = histfit(peaks_detected_length_list_zero,100,'gamma'); hold on;grid on;
% t2 = histfit(peaks_detected_length_list_acute,200,'gamma'); hold on;grid on;
% t1(1).FaceColor = [0, 0.4470, 0.7410];
% t1(1).FaceAlpha = 0.7;
% t1(1).EdgeColor = [0 0 0];
% t1(2).Color = 'b';
% t1(2).LineStyle = '-.';
%
% t2(1).FaceColor = [0.8500, 0.3250, 0.0980];
% t2(1).FaceAlpha = 0.7;
% t2(2).Color = 'r';
% xlim([0,1]);
% legend('\theta\approx0^\circ','\gamma fit to \theta\approx0^\circ','\theta\approx45^\circ','\gamma fit to \theta\approx45^\circ');
% ax = gca;
% ax.FontSize = 14;
% xlabel('Ratio of SP and PT');
% ylabel('Frequency');

area_ratio_list_zero(peaks_detected_length_list_zero<0.7) = [];
peaks_detected_length_list_zero(peaks_detected_length_list_zero<0.7) = [];

area_ratio_list_acute(peaks_detected_length_list_acute<0.7) = [];
peaks_detected_length_list_acute(peaks_detected_length_list_acute<0.7) = [];

peaks_detected_length_list_zero(area_ratio_list_zero>1) = [];
area_ratio_list_zero(area_ratio_list_zero>1) = [];

peaks_detected_length_list_acute(area_ratio_list_acute>1) = [];
area_ratio_list_acute(area_ratio_list_acute>1) = [];


min_datalength = min(length(area_ratio_list_zero),length(area_ratio_list_acute));

area_ratio_list_zero = area_ratio_list_zero(1:min_datalength);
peaks_detected_length_list_zero = peaks_detected_length_list_zero(1:min_datalength);
area_ratio_list_acute = area_ratio_list_acute(1:min_datalength);
peaks_detected_length_list_acute = peaks_detected_length_list_acute(1:min_datalength);

area_ratio_list_zero_bak = area_ratio_list_zero;
area_ratio_list_acute_bak = area_ratio_list_acute;



% pd1 = makedist('HalfNormal','mu',0,'sigma',0.255);
% x = 0:0.001:1;
% pdf1 = pdf(pd1,x);
% pdf1 = pdf1./max(pdf1);
%
% figure;
% plot(x,pdf1,'r','LineWidth',2)
% hold on;grid on;
% plot(x,1-pdf1,'b','LineWidth',2); hold on;grid on;
% xlabel('f');
% ylabel('Probability');
% ax = gca;
% ax.FontSize = 14;
% xlim([0,1]);
% index = find(pdf1<0.5,1,'first');
% disp(['Parameter at 0.5 probability: ', num2str(x(index))]);
% legend('P(\theta=0^\circ)','P(\theta\neq0^\circ)');

%
% figure;
% t1 = histfit(area_ratio_list_zero,50,'gamma'); hold on;grid on;
% pd1 = fitdist(area_ratio_list_zero','gamma');

% reverse halfnormal
% area_ratio_list_acute_reverse = 1-area_ratio_list_acute;

% t2 = histfit(area_ratio_list_acute,100,'gamma'); hold on;grid on;
% pd2 = fitdist(area_ratio_list_acute','gamma');

figure;
hzero = histogram(area_ratio_list_zero,50,'Normalization','pdf','BinWidth',0.01); hold on;grid on;
hacute = histogram(area_ratio_list_acute,50,'Normalization','pdf','BinWidth',0.01); hold on;grid on;
xlim([0,1]);
%
%
% figure;

% x1 = 0:0.001:1.5;
% y1 = gampdf(x1, pd1.a, pd1.b);
% y2 = gampdf(x1, pd2.a, pd2.b);
% gamma1 = [pd1.a,pd1.b];
% gamma2 =[pd2.a, pd2.b];
% gamma = [gamma1, gamma2];
% save('results/gamma_parameters.mat','gamma');

% t1(1).FaceColor = [0, 0.4470, 0.7410];
% t1(1).FaceAlpha = 0.7;
% t1(1).EdgeColor = [0 0 0];
% t1(2).Color = 'b';
% t1(2).LineStyle = '-.';
%
%
% t2(1).FaceColor = [0.8500, 0.3250, 0.0980];
% t2(1).FaceAlpha = 0.7;
% t2(2).Color = 'r';
% legend('\theta\approx0^\circ','\gamma fit to \theta\neq0^\circ','\theta\approx45^\circ','\gamma fit to \theta\approx45^\circ');
legend('\theta\approx0^\circ','\theta\approx45^\circ');
ax = gca;
ax.FontSize = 14;
xlabel('f');
ylabel('Frequency');



% figure;
% plot(x1,y1,'b--','LineWidth',2); hold on;grid on;
% plot(x1,y2,'r-','LineWidth',2); hold on;grid on;
% xlabel('f');
% ylabel('Frequency');
% legend('\theta\approx0^\circ','\theta\neq0^\circ','\gamma fit to \theta\approx0^\circ','\gamma fit to \theta\neq0^\circ');
% ax = gca;
% ax.FontSize = 14;
% xlim([0,1]);


true_labels = [zeros(1,length(area_ratio_list_zero_bak)) ones(1,length(area_ratio_list_acute_bak))];
%Identify Classification threshold
threshold_list = 0:0.0005:0.8;
class1_correct = zeros(1,length(threshold_list));
class2_correct = zeros(1,length(threshold_list));

for th = 1:length(threshold_list)
    threshold = threshold_list(th);
    feature_vector  = [area_ratio_list_zero_bak , area_ratio_list_acute_bak];
    predicted_labels = feature_vector>threshold;
    C = confusionmat(true_labels,double(predicted_labels));
    class1_correct(th) = C(1,1);
    class2_correct(th) = C(2,2);
end

total_accuracy = class1_correct +  class2_correct;
[~,max_ind] = max(total_accuracy);
threshold = threshold_list(max_ind);
disp(['Threshold: ',num2str(threshold)]);

% Angle classification accuracy for the 2 angles
feature_vector  = [area_ratio_list_zero_bak , area_ratio_list_acute_bak];
predicted_labels = feature_vector>threshold;
true_labels = true_labels + 1;
predicted_labels = predicted_labels + 1;
class_labels = {'zero','acute'};
predicted_class = class_labels(predicted_labels);
true_class = class_labels(true_labels);
figure;
cm = confusionchart(true_class, predicted_class, 'Normalization','row-normalized');
classification_accuracy = (cm.NormalizedValues(1,1) +  cm.NormalizedValues(2,2))/2;
disp(['Classification accuracy: ' , num2str(classification_accuracy)]);
ax = gca;
ax.FontSize = 20;
sortClasses(cm,["zero","acute"])

% [C] = confusionmat(true_labels,double(predicted_labels));
% figure;
% cm = confusionchart(C,'Normalization','row-normalized','FontSize',12,order);
% classification_accuracy = (cm.NormalizedValues(1,1) +  cm.NormalizedValues(2,2))/2;
% disp(['Classification accuracy: ' , num2str(classification_accuracy)]);

% figure;
% subplot(1,2,1);
% histogram2(area_ratio_list_zero',peaks_detected_length_list_zero','Normalization','probability'); hold on;grid on;
% xlabel('r')
% ylabel('f')
% ax = gca; ax.FontSize = 12;
% subplot(1,2,2);
% histogram2(area_ratio_list_acute',peaks_detected_length_list_acute','Normalization','probability'); hold on;grid on;
% xlabel('r')
% ylabel('f')
% ax = gca; ax.FontSize = 12;

x = 0:0.0005:1;
overall_min = 1;
for s1 = 0.1:0.001:0.3
    for s2 = 0.1:0.001:0.6
        pd1 = makedist('HalfNormal','mu',0,'sigma',s1);
        pd2 = makedist('Normal','mu',1,'sigma',s2);
        
        pdf1 = pdf(pd1,x);
        pdf2 = pdf(pd2,x);
        % Find intersection point
        ydiff = abs(pdf1-2*pdf2);
        [min_val,min_ind] =  min(ydiff);
        if (abs(x(min_ind)-threshold))<overall_min
            overall_min = abs(x(min_ind)-threshold);
            min_s1 = s1;
            min_s2 = s2;
        end
    end
end

pd1 = makedist('HalfNormal','mu',0,'sigma',min_s1);
pd2 = makedist('Normal','mu',1,'sigma',min_s2);
pdf1 = pdf(pd1,x);
pdf2 = pdf(pd2,x);
figure;
plot(x,pdf1,'r','LineWidth',2);
hold on;
plot(x,2*pdf2,'b','LineWidth',2);
hold on;grid on;

factor = pdf1+ 2*pdf2;
normalized_pdf1 = pdf1./factor;
normalized_pdf2 = pdf2./factor;
figure;
plot(x,normalized_pdf1,'r','LineWidth',2);
hold on;
plot(x,2*normalized_pdf2,'b','LineWidth',2);
hold on;grid on;

s = [min_s1,min_s2];
save('results/angle_pdf_parameters_sf_3.mat','s');
