function [zz] = calculate_TRRS_MRC(CSI,refCSI,debug,y_upper_limit)

csi_testing = CSI;
csi_training = refCSI;

[Ntx, Nrx, ~,Nf] = size(csi_training);
for tx=1:1:Ntx
    for rx=1:1:Nrx
        d=(tx-1)*Nrx+rx;
        H1=squeeze(csi_training(tx,rx,:,:)).';
        H2=squeeze(csi_testing(tx,rx,:,:)).';
        H1 = H1./sqrt(sum(abs(H1).^2,2));
        H2 = H2./sqrt(sum(abs(H2).^2,2));
        [focus_mtx{d},focus_mtx_cpx{d}] = confusion_freq_array_fingerprint_v3(H1, H2);
    end
end

min_trrs_values = zeros(1,Ntx*Nrx);
denominator = 0;
figure;
for tx=1:1:Ntx
    for rx=1:1:Nrx
        d=(tx-1)*Nrx+rx;
        subplot(Ntx, Nrx,d);
        imagesc(focus_mtx{d}.^2);
        colormap jet;colorbar;caxis([y_upper_limit 1]);
        temp_matrix = focus_mtx{d}.^2;
        min_trrs_values(d) = min(temp_matrix(:));
        if (d==1)
            zz = (1-min_trrs_values(d))*temp_matrix;            
        else
            zz = zz + (1-min_trrs_values(d))*temp_matrix;
        end
        denominator = denominator + (1-min_trrs_values(d));
    end
end

zz = zz./denominator;

figure; 
imagesc(zz);colormap jet; colorbar; caxis([y_upper_limit 1]);
title('TRRS - MRC');
end