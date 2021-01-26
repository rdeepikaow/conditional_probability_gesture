function [character,probability_score,posterior_CP_matching] = gesture_classification_CP_4seg(angle_probabilities,matching_points_fractions)
% Code to read prior probabilities from excel sheet
excel_filename = 'prior_probabilities.xlsx';
excel_sheetname = 'AnglePrior4';
[num,txt,raw] = xlsread(excel_filename,excel_sheetname);
prior_CP_angles = num;
excel_sheetname = 'MPrior4';
[num,txt,raw] = xlsread(excel_filename,excel_sheetname);
prior_CP_matching = num(:,1:6);

% Convert each angle probabilities into two angle probabilities
prior_CP_probabilities_angles = zeros(5,8);
for i = 1:5
    prior_CP_probabilities_angles(i,1) = angle_probabilities(1,1)*angle_probabilities(1,2)*angle_probabilities(1,3);
    prior_CP_probabilities_angles(i,2) = angle_probabilities(1,1)*angle_probabilities(1,2)*angle_probabilities(2,3);
    prior_CP_probabilities_angles(i,3) = angle_probabilities(1,1)*angle_probabilities(2,2)*angle_probabilities(1,3);
    prior_CP_probabilities_angles(i,4) = angle_probabilities(1,1)*angle_probabilities(2,2)*angle_probabilities(2,3);
    prior_CP_probabilities_angles(i,5) = angle_probabilities(2,1)*angle_probabilities(1,2)*angle_probabilities(1,3);
    prior_CP_probabilities_angles(i,6) = angle_probabilities(2,1)*angle_probabilities(1,2)*angle_probabilities(2,3);
    prior_CP_probabilities_angles(i,7) = angle_probabilities(2,1)*angle_probabilities(2,2)*angle_probabilities(1,3);
    prior_CP_probabilities_angles(i,8) = angle_probabilities(2,1)*angle_probabilities(2,2)*angle_probabilities(2,3);
end

% Convert matching point probabilities 
posterior_CP_matching = zeros(1,5);
for c = 1:5
    posterior_CP_matching(c) = 1- ((MP_function(abs(matching_points_fractions(1)-prior_CP_matching(c,1)))+MP_function(abs(matching_points_fractions(2)-prior_CP_matching(c,2))))/2+...
        +(MP_function(abs(matching_points_fractions(3)-prior_CP_matching(c,3)))+MP_function(abs(matching_points_fractions(4)-prior_CP_matching(c,4))))/2+...
        (MP_function(abs(matching_points_fractions(5)-prior_CP_matching(c,5)))+MP_function(abs(matching_points_fractions(6)-prior_CP_matching(c,6))))/2)/3;
end
posterior_CP_matching = posterior_CP_matching./sum(posterior_CP_matching);
probability_score = zeros(1,5);
for c = 1:size(probability_score,2)
    probability_score(c) = sum(prior_CP_probabilities_angles(c,:).*prior_CP_angles(c,:));
    probability_score(c) = probability_score(c).*posterior_CP_matching(c);
end

probability_score = probability_score./sum(probability_score);
disp(['Probability scores: ',num2str(probability_score)]);
[~,max_ind] = max(probability_score);
character_list = {'A','O','R','J','M'};
character = character_list{max_ind};
disp(['Character: ',character]);
end


