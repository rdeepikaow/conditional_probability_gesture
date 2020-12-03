function [startSegments,endSegments] = MotionSegmentation5300(motionStats,debug,params)
%% Returns the start and end index for line segments in a gesture

if (debug)
    figure; plot(motionStats);hold on;grid on;
end
L = length(motionStats);
w = params.min_ms_win;
potentialMin = [];
for i = 1:L
    if (motionStats(i)<=min([motionStats(max(1,i-w):max(1,i-1)) motionStats(min(i+1,L):min(i+w,L))]))
        if (motionStats(i)<params.turn_loc_ms_upper_th)
            potentialMin = [potentialMin i];
        elseif isempty(potentialMin)
            potentialMin = [i];
        end
    end
    
    
end
if (debug)
plot(potentialMin,motionStats(potentialMin),'ro'); hold on;grid on;
end

%% Find distinct points
diffPotentialMin = diff(potentialMin);
distinctPotentialMin = find(diffPotentialMin>params.distinct_turns_min_sample_diff);

endSegments = potentialMin(distinctPotentialMin+1);
startSegments = potentialMin(distinctPotentialMin);

%% Check if there is a significant motion in that segment
removeIdx = [];
for i = 1:length(endSegments)
    if max(motionStats(startSegments(i):endSegments(i)))<params.sig_motion_lower_th
        removeIdx = [removeIdx i];
    end
end

endSegments(removeIdx) = [];
startSegments(removeIdx) = [];
if (debug)
    plot(startSegments,motionStats(startSegments),'go'); hold on;grid on;
    plot(endSegments,motionStats(endSegments),'bo'); hold on;grid on;
end

if (length(endSegments)==length(startSegments))
    nsegments = length(endSegments);
    disp(['Number of segments = ',num2str(nsegments)]);
else
    disp(['Cannot find number of segments']);
end

