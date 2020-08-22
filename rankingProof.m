%%
sd = "/home/srikanth/Documents/sednn-master/mixture2clean_dnn/full_data/train_speech/";
speechDir = dir(sd);
n1 = length(speechDir);
sd = speechDir(3).folder;

%nSegs = 300;
gaps = zeros(20000,1);

L = 160;
N = 512;
fs = 16000;
Ithres = 5;

ntrials = 50;
VAD_cst_param = vadInitCstParams;

Pall = [1,2,3,5];
SNRall = [0,10,20];
muall = [0.01,0.1,1];

results = cell(length(Pall),length(SNRall),length(muall));

tic
for SNR_ind = 1:length(SNRall)
    SNR = SNRall(SNR_ind);   
    for mu_ind = 1:length(muall)
        mu = muall(mu_ind);
        if(SNR == 0 || mu == 0.01)
            nSegs = 100;
        else
            nSegs = 50;
        end
        outputs = zeros(length(Pall),ntrials,2);
        parfor trial = 1:ntrials
            a=[];
            for i =1:nSegs
                tempAudio = audioread(strcat(sd,'/', speechDir(randi([3,n1])).name));
                a = [a;tempAudio;gaps];
            end
            refOut = a;
            h = hsample(N);
            mixedOut = filter(h,1,refOut);
            nse= randn(length(mixedOut),1);
            %enerRatio = mean(nse(nse~=0).^2)/mean(mixedOut(mixedOut~=0).^2);
            enerRatio = mean(nse.^2)/mean(mixedOut.^2);
            beta = sqrt(10^(SNR/10) * enerRatio);
            mixedOut = mixedOut + (1/beta)*nse;

            farSpeechSrc   = dsp.SignalSource('Signal',refOut,'SamplesPerFrame',L);
            micSignalSrc    = dsp.SignalSource('Signal',mixedOut,'SamplesPerFrame',L);

            AECfilter = dsp.LMSFilter(N,'Method','Normalized LMS','StepSize',mu, 'WeightsOutput', 'last'); 

            ref = []; mic = [];
            ref.vadState.biquad = [];
            mic.vadState.biquad = [];
            for i = 1:10
                [ref.vadStatus, ref.vadState] = vad729custom(0.01*randn(80,1),ref.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(0.0001*randn(80,1),mic.vadState, VAD_cst_param);
            end

            temperr=[]; tempyfilt = [];hMat=[];
            %tic
            while(~isDone(farSpeechSrc))
                farSpeech = farSpeechSrc();
                micSignal = micSignalSrc(); 
                %VAD trackers
                [ref.vadStatus, ref.vadState] = vad729custom(downsample(farSpeech,2),ref.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(downsample(micSignal,2),mic.vadState, VAD_cst_param);

                if(ref.vadStatus == 0 || mic.vadStatus == 0)
                    %fprintf('Silence \n');
                else
                    [yfilt, e, w] =AECfilter(farSpeech,micSignal);
                    hMat = [hMat,w];
                end
            end
            %toc

            tempRes = zeros(length(Pall),2);
            corrMet = corr( abs(hMat), abs(h'), 'Type', 'Pearson' );
            [~,hNew] = sort(hMat, 1, 'descend');
            [~,h2] = sort(h, 'descend');
            setB = h2(1); %Get the peak location
            for P_ind = 1:length(Pall) %Top-P locations
                P = Pall(P_ind);
                % Compare Top-P locations
                setA = sort(hNew(1:P,:), 1);
                

                C = zeros(size(setA,2),1);
                for i = 1:size(setA,2)
                    C(i) = length(intersect(setA(:,i), setB));
                end
                %C = C/P; 

                %{
                plot(C, 'b'); hold on; plot(corrMet, 'r'); ylim([0,1.01]);
                waitforbuttonpress;
                close all;
                %}


                C = (C==1);

                a = 0;
                for i = 1:size(C)
                    if(C(i) == 1)
                        a = a+1;
                        if(a==Ithres)
                            break;
                        end
                    else
                        a = 0;
                    end
                end

                a = find(corrMet > 0.95);
                %max(a)
                if(~isempty(a))
                    a=a(1);
                else
                    a = 1.5*length(C); %the extra 0.5 term is the penalty for not converging
                    fprintf('Didnt converge ...%d \n', length(C));
                    
                    %Store both as nan
                    %i = nan;
                    %a = nan;
                    %continue; %Don't store this trial val
                end
                tempRes(P_ind,:) = [i, a];
                %fprintf('\n %d, %d, %d, %d, %d, %d \n', i, a, P, SNR, mu, trial);
            end
            outputs(:,trial,:) = tempRes;
        end
        
        for P_ind = 1:length(Pall)
            P = Pall(P_ind);
            results{P_ind, SNR_ind, mu_ind} = [floor(mean( squeeze(outputs(P_ind,:,:)) ,1)), P, SNR, mu];
            %[floor(mean( squeeze(outputs(P_ind,:,:)) ,1)), P, SNR, mu]
            fprintf('\n %d, %d, P = %d, SNR = %d, mu = %d \n', floor(mean( squeeze(outputs(P_ind,:,:)) ,1)), P, SNR, mu)
        end
        
    end
end
toc





