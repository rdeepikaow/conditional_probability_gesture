classdef GestureConstruction
    % Contains pre-processing and post-processing methods for gesture
    % construction using 5GHz WiFi
    
    properties
        sf_ ;
        display_min_ ;
        outlier_removal_offset_ ;
        outlier_removal_th_ ;
        debug_;
        similarity_matrix_;
        similarity_matrix_smoothed_;
        filename_;
        matfilename_;
        correlation_;
        similarity_method_;
        crop_index_;
        num_useful_subcarriers_;
        motion_statistics_;
    end
    
    methods
        function obj = GestureConstruction(params)
            obj.sf_ = params.sf;
            obj.display_min_ = params.display_min;
            obj.outlier_removal_offset_ = params.outlier_removal_offset;
            obj.outlier_removal_th_ = params.outlier_removal_th;
            obj.debug_ = params.debug;
            obj.debug_ = params.debug;
            obj.filename_ = params.filename;
            obj.matfilename_ = params.matfilename;
            obj.similarity_method_ = params.similarity_method;
            obj.crop_index_ = params.crop_index;
            obj.num_useful_subcarriers_ = params.num_useful_subcarriers;
            obj.motion_statistics_ = [];
        end
        
        function obj = preprocess(obj,filename)
            data = load(filename);
            if (obj.crop_index_==-1)
                if (contains(filename,'phase_comp.mat'))
                    CSI = data.CSI_mtx(:,:,:,1:obj.sf_:end);
                else
                    CSI = data.csi_trace.csi(:,:,:,1:obj.sf_:end);
                    CSI = abs(CSI);
                end
            else
                if (contains(filename,'phase_comp.mat'))
                    CSI = data.CSI_mtx(:,:,:,1:obj.sf_:obj.crop_index_);
                else
                    CSI = data.csi_trace.csi(:,:,:,1:obj.sf_:obj.crop_index_);
                    CSI = abs(CSI);
                end
            end
            
            %% CSI normalized and vectorized - Thresholding outlier removal
            %             [~, ~, Ns,Nreal] = size(CSI);
            %             chnnorm = sqrt(sum(abs(CSI).^2, 3));
            %             csi_normalized = CSI ./ (repmat(chnnorm,[1,1,Ns,1]) + eps);
            %             csi_normalized(isnan(CSI) | isinf(CSI)) = 0;
            %
            %             csi_normalized = permute(csi_normalized, [3, 1, 2,4]);
            %             dimension_csi = size(csi_normalized);
            
            %% CSI normalized and vectorized - outlier removal end
            %             CSI = abs(CSI);
            if (contains(filename,'phase_comp.mat'))
                y_upper_limit = 0.95;
            end
            if (strcmp(obj.similarity_method_,'trrs'))
                obj.similarity_matrix_ = calculate_TRRS(CSI,CSI,obj.debug_,y_upper_limit);
                outliers =[];
                Nll = size(obj.similarity_matrix_,1);
                for i = 1:Nll-obj.outlier_removal_offset_
                    if (obj.similarity_matrix_(i,i+obj.outlier_removal_offset_)<obj.outlier_removal_th_)
                        outliers = [outliers i];
                    end
                    if (obj.similarity_matrix_(i,i)~=1)
                        outliers = [outliers i];
                    end
                end
%                 outliers = [];
                obj.similarity_matrix_(:,outliers)=[];
                obj.similarity_matrix_(outliers,:)=[];
            end
            if (strcmp(obj.similarity_method_ , 'correlation_phaseboost'))
                obj.similarity_matrix_ = calculate_correlation(obj.matfilename_,1,obj.num_useful_subcarriers_,obj.crop_index_,obj.sf_,obj.debug_);
            end
            if (strcmp(obj.similarity_method_,'correlation'))
                obj.similarity_matrix_ = calculate_correlation(obj.matfilename_,0,obj.num_useful_subcarriers_,obj.crop_index_,obj.sf_,obj.debug_);
            end
            if (obj.debug_==1)
                figure; imagesc(obj.similarity_matrix_);colorbar;caxis([obj.display_min_ 1]);
                xlabel('CSI sample index'); ylabel('CSI sample index');title('Correlation Matrix');
                ax = gca; ax.FontSize = 14;
            end
             obj.motion_statistics_ = MotionStatistics5300_updated(obj.matfilename_,0,obj.num_useful_subcarriers_,obj.sf_,obj.debug_,outliers);
        end
    end
end

