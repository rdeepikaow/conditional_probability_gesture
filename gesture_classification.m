function character = gesture_classification(angle_probabilities,matching_points_fractions)
% Code to read prior probabilities from excel sheet
excel_filename = 'prior_probabilities.xlsx';
excel_sheetname = 'Prior';
[num,txt,raw] = xlsread(excel_filename,excel_sheetname);
% Search all the rows with N
N = 3;
switch N
    case 3
        search_column = num(:,1);
        search_row_indices = find(search_column==1)+1;
        search_col_indices_angles = [4:9];
        search_col_indices_matching = 17;
        probability_score = zeros(1,length(search_row_indices));
        for c = 1:length(search_row_indices)
            conditional_probabilities_angles = num(search_row_indices(c)-1,search_col_indices_angles).*reshape(angle_probabilities,1,(N-1)*3);
            matching_points = raw(search_row_indices(c),search_col_indices_matching);
            matching_points = matching_points{1};
            if length(matching_points)>2
                matching_points_first = extractBetween(matching_points,'(',',');
                matching_points_first = str2num(matching_points_first{1});
                matching_points_second = extractBetween(matching_points,',',')');
                matching_points_second = str2num(matching_points_second{1});
                matching_point_probability = 1-(abs(matching_points_first-matching_points_fractions(search_col_indices_matching-16,1))+...
                    abs(matching_points_second-matching_points_fractions(search_col_indices_matching-16,2)))/2;
            else
                matching_point_probability = 0;
            end
            conditional_probability_angles = reshape(conditional_probabilities_angles,3,N-1);
            for i = 1:N-1
                conditional_probability_angles(:,i) = conditional_probability_angles(:,i)./sum(conditional_probability_angles(:,i));
            end
            total_probability = 0;
            for i = 1:3
                for j = 1:3
                    total_probability = total_probability + conditional_probability_angles(i,1)*conditional_probability_angles(j,2);
                end
            end
            if (matching_point_probability<0.5)
                total_probability = total_probability*0.5;
            else
                total_probability = total_probability*matching_point_probability;
            end
            probability_score(c) = total_probability; 
        end
        disp(raw(search_row_indices,1));
        disp(['Probability scores: ',num2str(probability_score)]);
        [~,max_ind] = max(probability_score);
        character = raw(search_row_indices(max_ind),1);        
    case 4
        search_column = num(:,2);
        search_row_indices = find(search_column==1)+1;
        search_col_indices = [5:13, 17, 18, 20];
    case 5
        search_column = num(:,3);
        search_row_indices = find(search_column==1)+1;
        search_col_indices = [5:22];
end


