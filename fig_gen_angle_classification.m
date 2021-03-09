% Code to verify the probability scores for the turn angle estimation.
clc; clear all; close all;
% for f = 0:0.001:1
%     for rho = 0:0.001:1
%         total_sum = sgn(f-0.5)*(1-rho)+sgn(f-0.2)*sgn(0.5-eps-f)+sgn(f-0.5)*rho + sgn(0.2-eps-f);
%         if total_sum~=1
%             disp(['f: ',num2str(f),' rho: ', num2str(rho), ' total_sum: ', num2str(total_sum)]);
%         end
%     end
% end

%% Zero degree turn

f = 0:0.001:1;
rho = 0:0.001:1;
pgraph = zeros(length(f),length(rho));
for fi = 1:length(f)
    for rhoi = 1:length(rho)
        pgraph(fi, rhoi) = sgn(f(fi)-0.5)*(1-rho(rhoi));
    end
end

figure;
imagesc(pgraph);colormap jet; colorbar;caxis([0 1]);

%% Acute angle turn

pgraph = zeros(length(f),length(rho));
for fi = 1:length(f)
    for rhoi = 1:length(rho)
        pgraph(fi, rhoi) = sgn(f(fi)-0.2)*sgn(0.5-eps-f(fi))+ sgn(f(fi)-0.5)*rho(rhoi) ;
    end
end

figure;
imagesc(pgraph);colormap jet; colorbar;caxis([0 1]);

%% Right angle turn

pgraph = zeros(length(f),length(rho));
for fi = 1:length(f)
    for rhoi = 1:length(rho)
        pgraph(fi, rhoi) =  sgn(0.2-eps-f(fi));
    end
end

figure;
imagesc(pgraph);colormap jet; colorbar;caxis([0 1]);

function out = sgn(difference)
if (difference>=0)
    out = 1;
else
    out = 0;
end
end