function [mixedOut,refOut,nearOut] = DTDdataSampler(speechDir, noiseDir, h, nSegs, NFR, SNR, noiseType)

    n1 = length(speechDir);
    sd = speechDir(3).folder;
    a=[]; b=[];
    gaps = zeros(2000,1);
    for i =1:2*nSegs-1
        tempAudio = audioread(strcat(sd,'/', speechDir(randi([3,n1])).name));
        if(mod(i,2)==1) %far end active
            a = [a;tempAudio;gaps];
            b = [b;zeros(length(tempAudio),1);gaps];
        else
            b = [b;tempAudio;gaps];
            a = [a;zeros(length(tempAudio),1);gaps];
        end
    end
    refOut = a;
    a = filter(h,1,a);
    %Creating doubletalk
    k = randi([200000,350000]);
    b = circshift(b,k);
    b(1:k) = 0;
    %enerRatio = mean(a(a~=0).^2)/mean(b(b~=0).^2);
    enerRatio = mean(a.^2)/mean(b.^2);
    alpha = sqrt(10^(NFR/10) * enerRatio);

    mixedOut = (1/alpha)*a + b;
    nearOut = b;


    if(noiseType == 1) 
        %Non-stationary noise
        n2 = length(noiseDir);
        nd = noiseDir(3).folder;
        tempAudio = audioread(strcat(nd,'/', noiseDir(randi([3,n2])).name));
        nse = repmat(tempAudio, [floor(length(mixedOut)/length(tempAudio)),1]);
        nse = [nse;zeros(length(mixedOut) - length(nse),1)];

    else
        nse= randn(length(mixedOut),1);
    end

    %enerRatio = mean(nse(nse~=0).^2)/mean(mixedOut(mixedOut~=0).^2);
    enerRatio = mean(nse.^2)/mean(mixedOut.^2);
    beta = sqrt(10^(SNR/10) * enerRatio);
    mixedOut = mixedOut + (1/beta)*nse;

    %Add rir or reverb later for simulation studies
end