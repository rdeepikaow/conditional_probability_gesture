clc; clear all; close all;
groundTruth_indices = [ones(1,100) ones(1,100).*2 ones(1,100).*3];
predicted_indices = [ones(1,80) ones(1,20).*2 ones(1,10).*2 ones(1,90)*1 ones(1,100).*3];

class_labels = {'\theta','\theta\approx45^\circ', '\theta\approx90^\circ'};

figure;
cm = confusionchart(class_labels(groundTruth_indices),class_labels(predicted_indices));
figure;
cm_indices = confusionchart(groundTruth_indices,predicted_indices,...
    'Normalization','row-normalized');
