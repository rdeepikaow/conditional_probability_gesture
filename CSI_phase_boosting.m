classdef CSI_phase_boosting < handle
    %CSI contains basic operations related to CSI values
    %    basic operations of CSI, such as normalization, interpolation,
    %    fingerprinting, etc.
    
    properties
        csi_raw_                        % raw csi with dimension tx * rx * subcarr * packet
        csi_normalized_                 % normalized the power per link
        csi_fingerprint_                % csi fingerprint based on folded method
        csi_matrix_                     % squeeze the normalized csi to a 2d matrix
        csi_power_matrix_               % 2d matrix for csi power response
        csi_power_matrix_interp_        % 2d matrix for interpolated csi power response
        timestamp_raw_                  % raw timestamp in seconds
        timestamp_                      % filtered timestamp (remove very positive/negative values)
        sampling_rate_                  % actual sampling rate of CSI packets
        carrier_frequency_              % carrier frequency of transmitted signal
        num_packets_                    % num of packets in total
        num_subcarriers_                % num of subcarriers per link
        num_links_                      % num of links
    end
    
    
    properties (Constant)
        speed_of_light_ = 3.0e8;
    end
    
    
    properties (Dependent)
        wavelength_
    end
    
    
    methods
        
        function obj = CSI_phase_boosting(csi_raw, timestamp, downsample_subcarrier, downsample_time, carrier_frequency,phase_boosting)
            % initilize a CSI object
            obj.csi_raw_ = CSI_phase_boosting.DownSample(csi_raw, downsample_subcarrier, downsample_time);
            obj.csi_normalized_ = CSI_phase_boosting.Normalize(obj.csi_raw_);
            %       obj.csi_fingerprint_ = CSI.Fingerprint(obj.csi_raw_);
            
            %% Modified code to include phase in correlation calculation
            if (phase_boosting)
                csi_processed = zeros(size(obj.csi_normalized_));
                csi_ratio21 = obj.csi_normalized_(:, 2, :,:) ./ (obj.csi_normalized_(:, 1, :,:) + eps);
%                 csi_ratio31 = obj.csi_normalized_(:, 3, :,:) ./ (obj.csi_normalized_(:, 1, :,:) + eps);
                csi_processed(:,1,:,:) = (obj.csi_normalized_(:,1,:,:));
%                 csi_processed(:,2,:,:) = abs(obj.csi_normalized_(:,2,:,:)).*exp(1i.*angle(csi_ratio21));
%                 csi_processed(:,3,:,:) = abs(obj.csi_normalized_(:,3,:,:)).*exp(1i.*angle(csi_ratio31));
                obj.csi_normalized_ = csi_processed;
                obj.csi_normalized_ = CSI_phase_boosting.Normalize(obj.csi_normalized_);
            end
            %% Modified code to include phase in correlation calculation - End
            
            obj.csi_matrix_ = CSI_phase_boosting.Matrix(obj.csi_normalized_);
            obj.csi_power_matrix_ = abs(obj.csi_matrix_).^2;
            obj.num_packets_ = size(obj.csi_raw_, 4);
            obj.num_subcarriers_ = size(obj.csi_raw_, 3);
            obj.num_links_ = size(obj.csi_raw_, 1) * size(obj.csi_raw_, 2);
            obj.timestamp_raw_ = reshape(timestamp, 1, length(timestamp));
%             [obj.timestamp_, obj.sampling_rate_] = CSI.Time(obj.timestamp_raw_, downsample_time);
            if nargin == 5
                obj.carrier_frequency_ = carrier_frequency;
            end
        end
        
        function value = get.wavelength_(obj)
            % update the wavelength of the transmitted signal
            value = obj.speed_of_light_ / obj.carrier_frequency_;
        end
        
    end
    
    
    methods (Static)
        
        function csi_normalized = Normalize(csi)
            % normalize the power of CSI for each link
            if ndims(csi) == 4
                [~, ~, Ns, ~] = size(csi);
                chnnorm = sqrt(sum(abs(csi).^2,3));
                csi_normalized = csi./repmat(chnnorm,[1,1,Ns,1]);
                csi_normalized(isnan(csi) | isinf(csi)) = 0;
            else
                error('The number of the dimension of the CSI matrix is not 4!')
            end
        end
        
        function csi_fingerprint = Fingerprint(csi)
            % normalized csi fingerprint using folded method
            csi_fingerprint = CSI_phase_boosting.Normalize(csi( : , : , 1 : end/2, : ) .* csi( : , : , end : -1 : end/2+1, : ));
        end
        
        function csi_matrix = Matrix(csi)
            dimension_csi = size(csi);
            csi_matrix = zeros(prod(dimension_csi(1:3)),dimension_csi(4));
            for tx = 1:dimension_csi(1)
                for rx = 1:dimension_csi(2)
                    link_index = (rx-1)*dimension_csi(1)+tx;
                    for subca = 1:dimension_csi(3)
                        csi_matrix((link_index-1)*dimension_csi(3)+subca,:) = reshape(squeeze(csi(tx,rx,subca,:)),[1,length(csi(tx,rx,subca,:))]);
                    end
                end
            end
        end
        
        function [timestamp, sampling_rate] = Time(timestamp_raw, downsample_time)
            timestamp_raw = timestamp_raw(1 : downsample_time : end) / 10^6;
            timestamp_raw_diff = diff(timestamp_raw);
            timestamp_raw_diff_pos = max(timestamp_raw_diff, 0);  % remove negative timestamp
            sampling_rate_tmp = 1 / mean(timestamp_raw_diff_pos);
            timestamp_raw_diff_pos = min(timestamp_raw_diff_pos, 4 / sampling_rate_tmp);  % remove very large timestamp
            sampling_rate = 1 / mean(timestamp_raw_diff_pos);
            timestamp = [0, cumsum(timestamp_raw_diff_pos)];
        end
        
        function csi = DownSample(csi_raw, downsample_subcarrier, downsample_time)
            switch downsample_subcarrier
                case 1
                    csi = csi_raw( :, :, :, 1 : downsample_time : end);
                case 2
                    subcarrier_index = [-58:2:-2 2-3:2:58-3] + 59;
                    csi = csi_raw( :, :, subcarrier_index, 1 : downsample_time : end);
                case 4
                    subcarrier_index = [-58:4:-2 2-3:4:58-3] + 59;
                    csi = csi_raw( :, :, subcarrier_index, 1 : downsample_time : end);
                otherwise
                    error('Unknown downsample number for subcarriers!')
            end
        end
        
    end
    
    
end

