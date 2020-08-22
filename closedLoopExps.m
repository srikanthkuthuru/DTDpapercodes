%%
sd = "/home/srikanth/Documents/sednn-master/mixture2clean_dnn/full_data/train_speech/";
speechDir = dir(sd);
nd = "/home/srikanth/Documents/sednn-master/mixture2clean_dnn/full_data/train_noise/";
noiseDir = dir(nd);

n1 = length(speechDir);
sd = speechDir(3).folder;

%nSegs = 300;
gaps = zeros(20000,1);

L = 160;
M = L;
N = 512;
fs = 16000;
Ithres = 10;
nccLen = 3*L;
T = 0.7;
Tcl = 0.98;
Tncc = 0.98;
P = 5;

ntrials = 25;
VAD_cst_param = vadInitCstParams;

noiseType = 0;
SNRall = [100];
NFRall = [-10,-5,0,5,10,15,20];
mu = 0.5;


RecallResults = zeros([3,length(SNRall),length(NFRall)]);
PrecisionResults = zeros([3,length(SNRall),length(NFRall)]);

tic
for SNR_ind = 1:length(SNRall)
    for NFR_ind = 1:length(NFRall)
        SNR = SNRall(SNR_ind);
        NFR = NFRall(NFR_ind);
        %mu = muall(mu_ind);
        if(SNR == 0 || mu == 0.01)
            nSegs = 100;
        else
            nSegs = 50;
        end
        parfor trial = 1:ntrials
            c1 = 0; c2 = 0; c3 = 0; cStatus = 0;
            h = hsample(N);
            hLR = fliplr(h);
            [mixedOut,refOut,nearOut] = DTDdataSampler(speechDir, noiseDir, h, nSegs, NFR, SNR, noiseType);
            %figure(1); subplot(311);plot(refOut);subplot(312);plot(nearOut);subplot(313);plot(mixedOut);
            %sound(mixedOut, fs);
            %waitforbuttonpress;

            SpkrSrc   = dsp.SignalSource('Signal',refOut,'SamplesPerFrame',M);
            micSrc    = dsp.SignalSource('Signal',mixedOut,'SamplesPerFrame',M);
            nearSrc    = dsp.SignalSource('Signal',nearOut,'SamplesPerFrame',M);
            
            ybuff = dsp.AsyncBuffer(nccLen); write(ybuff, zeros(nccLen,1));
            yFbuff1 = dsp.AsyncBuffer(nccLen); write(yFbuff1, zeros(nccLen,1));
            yFbuff2 = dsp.AsyncBuffer(nccLen); write(yFbuff2, zeros(nccLen,1));
            pbuff = cell(P,1);
            for i=1:P
                pbuff{i} = dsp.AsyncBuffer(Ithres); write(pbuff{i}, zeros(Ithres,1));
            end

            lmsF1 =  customLMSfilt(N, L, mu);
            lmsF2 =  customLMSfilt(N, L, mu);
            lmsF31 =  customLMSfilt(N, L, mu);
            lmsF32 =  customLMSfilt(N, L, mu);

            %{
            AECScope1   = dsp.TimeScope(6, fs, ...
                        'LayoutDimensions', [6,1], ...
                        'TimeSpan', 30, 'TimeSpanOverrunAction', 'Scroll', ...
                        'BufferLength', 100*fs);

                    
            pk = 1; %Peak signal value
            AECScope1.ActiveDisplay = 1;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-pk pk];
            AECScope1.Title         = 'Far-End Speech Signal';

            AECScope1.ActiveDisplay = 2;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-pk pk];
            AECScope1.Title         = 'Near Signal';

            AECScope1.ActiveDisplay = 3;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-1 1];
            AECScope1.Title         = 'Mic signal';

            AECScope1.ActiveDisplay = 4;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-0.5 1.2];
            AECScope1.Title         = 'DTD output1';

            AECScope1.ActiveDisplay = 5;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-0.5 1.2];
            AECScope1.Title         = 'DTD output2';

            AECScope1.ActiveDisplay = 6;
            AECScope1.ShowGrid      = true;
            AECScope1.YLimits       = [-0.5 1.2];
            AECScope1.Title         = 'DTD output3';
            %}
            
            dtdValArr1 = [];
            dtdValArr2 = [];
            dtdValArr3 = [];
            refVAD = [];
            nearVAD = [];
            
            echoref = []; near = []; mic = [];
            echoref.vadState.biquad = [];
            near.vadState.biquad = [];
            mic.vadState.biquad = [];
            mic.prevCoefficients = randn([N,1]);
            for i = 1:10
                [echoref.vadStatus, echoref.vadState] = vad729custom(0.01*randn(80,1),echoref.vadState, VAD_cst_param);
                [near.vadStatus, near.vadState] = vad729custom(0.0001*randn(80,1),near.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(0.0001*randn(80,1),mic.vadState, VAD_cst_param);
            end

            %tic
            count1 = 0; count2 = 0; count3 = 0;
            while(~isDone(SpkrSrc))
                echoref.frame = SpkrSrc();
                mic.frame = micSrc();
                near.frame = nearSrc(); 
                
                
                %VAD trackers
                [echoref.vadStatus, echoref.vadState] = vad729custom(downsample(echoref.frame,2),echoref.vadState, VAD_cst_param);
                [near.vadStatus, near.vadState] = vad729custom(downsample(near.frame,2),near.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(downsample(mic.frame,2),mic.vadState, VAD_cst_param);

                
                if(echoref.vadStatus == 0 || mic.vadStatus == 0)
                    %fprintf('Silence \n');
                    c1 = 1;
                    c2 = 1;
                    c3 = 1;
                    dtdValArr1 = [dtdValArr1; 1*ones(M,1)]; 
                    dtdValArr2 = [dtdValArr2; 1*ones(M,1)];
                    dtdValArr3 = [dtdValArr3; 1*ones(M,1)];

                    refVAD = [refVAD; echoref.vadStatus*ones(M,1)];
                    nearVAD = [nearVAD; near.vadStatus*ones(M,1)];
                    
                else
                    if(near.vadStatus == 0 && cStatus == 0)
                        % Need to simulate open-loop method here - assume open loop works perfectly
                        % Here we are using unavailable information - nearEnd vad status
                        [~, ~, ~] = lmsF1.step(echoref.frame,mic.frame);
                        [~, ~, ~] = lmsF2.step(echoref.frame,mic.frame);
                        [~, ~, ~] = lmsF31.step(echoref.frame,mic.frame);
                        [yfilt, e, w] = lmsF32.step(echoref.frame,mic.frame);
                        
                        if(xcorr( abs(hLR) , abs(w), 0, 'coeff' ) > 0.9)
                            %After initial convergence - use only closed loop for update decisions -
                            %maintain 2 separate AECfilters
                            cStatus = 1;
                        else
                            cStatus = 0;
                            %fprintf('Not converged...%d \n', xcorr( abs(hLR) , abs(w), 0, 'coeff' )');
                        end
                        %figure(2);plot(w);
                        prevCheckpt1 = w;
                        prevCheckpt2 = w;
                        prevCheckpt3 = w;
                    elseif(near.vadStatus == 1 && cStatus == 0)
                        
                        [~, ~, ~] =lmsF1.filtStep(echoref.frame,mic.frame);
                        [~, ~, ~] =lmsF2.filtStep(echoref.frame,mic.frame);
                        [~, ~, ~] =lmsF31.filtStep(echoref.frame,mic.frame);
                        [yfilt, e, w] =lmsF32.filtStep(echoref.frame,mic.frame);
                    else
                        % cStatus is 1 here - so filters are converged - use closedloop DTD
                        
                        % Mic Buffer Update
                        read(ybuff, L); % Drop the first M samples
                        write(ybuff,mic.frame);
                        mic.buffer = peek(ybuff); % Get entire buffer 
                        
                        %------Closed loop DTD methods---------%
                        %---CLCC-----%
                        if(c1 > Tcl)
                            [yfilt, e, w] = lmsF1.step(echoref.frame,mic.frame);
                            count1 = count1 + 1;
                        else
                            lmsF1.Coefficients = prevCheckpt1;
                            count1 = 0;
                            [yfilt, e, w] = lmsF1.filtStep(echoref.frame,mic.frame);
                        end
                        if(count1 > 200)
                            prevCheckpt1 = lmsF1.Coefficients;
                            count1 = 0;
                        end
                        
                        % Predicted Signal Buffer Update
                        read(yFbuff1, L); % Drop the first M samples
                        write(yFbuff1,yfilt);
                        mic.Filtbuffer1 = peek(yFbuff1); % Get entire buffer 

                        sigy =  sqrt(mean(mic.buffer.^2));
                        sigy_ =  sqrt(mean(mic.Filtbuffer1.^2));
                        c1 = abs( mean(mic.buffer .* mic.Filtbuffer1))/(sigy * sigy_);


                        
                        %----NCC approx----%
                        if(c2 > Tncc)
                            [yfilt, e, w] = lmsF2.step(echoref.frame,mic.frame);
                            count2 = count2 + 1;
                        else
                            lmsF2.Coefficients = prevCheckpt2;
                            count2 = 0;
                            [yfilt, e, w] = lmsF2.filtStep(echoref.frame,mic.frame);
                        end
                        if(count2 > 200)
                            prevCheckpt2 = lmsF2.Coefficients;
                            count2 = 0;
                        end
                        % Predicted Signal Buffer Update
                        read(yFbuff2, L); % Drop the first M samples
                        write(yFbuff2,yfilt);
                        mic.Filtbuffer2 = peek(yFbuff2); % Get entire buffer 
                        
                        c2 = sqrt( mean(mic.buffer .* mic.Filtbuffer2) )/sigy;

                        
                        
                        %---FastPT-------%
                        [~, ~, w] = lmsF31.step(echoref.frame,mic.frame);
                            
                        %read(pbuff, 1); % Drop the first sample
                        %[~,maxLoc] = max(w);
                        %write(pbuff,maxLoc);
                        %c3 = (std(peek(pbuff)) == 0);
                        
                        
                        [~,wNew] = sort(w, 'descend');
                        setA = sort(wNew(1:P), 1);
                        c3 = 1;
                        for i = 1:P
                            read(pbuff{i}, 1);
                            write(pbuff{i},setA(i));
                            c3 = c3 * (std(peek(pbuff{i})) == 0);
                        end
                           
                        %fprintf('c1 = %d, c2 = %d, c3 = %d \n', c1, c2, c3);
                        %figure(3);plot(w);
                        
                        
                        
                        if(c3 == 1)
                            [yfilt, e, w] = lmsF32.step(echoref.frame,mic.frame);
                            count3 = count3 + 1;
                        else
                            lmsF32.Coefficients = prevCheckpt3;
                            count3 = 0;
                            [yfilt, e, w] = lmsF32.filtStep(echoref.frame,mic.frame);
                        end
                        if(count3 > 200)
                            prevCheckpt3 = lmsF32.Coefficients;
                            count3 = 0;
                        end
                        
                        %figure(4);plot(lmsF32.Coefficients);

                        dtdValArr1 = [dtdValArr1; c1*ones(L,1)]; 
                        dtdValArr2 = [dtdValArr2; c2*ones(L,1)];
                        dtdValArr3 = [dtdValArr3; c3*ones(L,1)];

                        refVAD = [refVAD; echoref.vadStatus*ones(L,1)];
                        nearVAD = [nearVAD; near.vadStatus*ones(L,1)];

                    end
                
                end
                
                %{
                AECScope1([echoref.frame, (pk-0.1)*echoref.vadStatus*(ones(M,1))],...
                    [near.frame, (pk-0.1)*near.vadStatus*(ones(M,1))],...
                    [mic.frame, (pk-0.1)*mic.vadStatus*(ones(M,1))],...
                    (c1>Tcl)*(ones(M,1)),...
                    (c2>Tncc)*(ones(M,1)),...
                    c3*(ones(M,1)));
                %}
                
            end
            %toc
            dtdTrue = (refVAD>T).*(nearVAD>T);
            dtdValArr1new = 1 - (dtdValArr1 > Tcl);
            dtdValArr2new = 1 - (dtdValArr2 > Tncc);
            dtdValArr3new = 1 - dtdValArr3;
            
            %[sum(and(dtdValArr1new,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr2new,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr3new,dtdTrue))/sum(dtdTrue)]
            
            %[sum(and(dtdValArr1new,dtdTrue))/sum(dtdValArr1new), sum(and(dtdValArr2new,dtdTrue))/sum(dtdValArr2new), sum(and(dtdValArr3new,dtdTrue))/sum(dtdValArr3new)]

            RtempResults(:, trial) = [sum(and(dtdValArr1new,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr2new,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr3new,dtdTrue))/sum(dtdTrue)];
            fprintf('\n SNR = %d, NFR = %d, trial = %d, Recalls = %d, %d, %d \n', SNR, NFR, trial, RtempResults(:, trial));
            
            PtempResults(:, trial) = [sum(and(dtdValArr1new,dtdTrue))/sum(dtdValArr1new), sum(and(dtdValArr2new,dtdTrue))/sum(dtdValArr2new), sum(and(dtdValArr3new,dtdTrue))/sum(dtdValArr3new)];
            fprintf('\n SNR = %d, NFR = %d, trial = %d, Precisions = %d, %d, %d \n', SNR, NFR, trial, PtempResults(:, trial));
        end
        RecallResults(1,SNR_ind,NFR_ind) = mean(RtempResults(1,:));
        RecallResults(2,SNR_ind,NFR_ind) = mean(RtempResults(2,:));
        RecallResults(3,SNR_ind,NFR_ind) = mean(RtempResults(3,:));
        
        PrecisionResults(1,SNR_ind,NFR_ind) = mean(PtempResults(1,:));
        PrecisionResults(2,SNR_ind,NFR_ind) = mean(PtempResults(2,:));
        PrecisionResults(3,SNR_ind,NFR_ind) = mean(PtempResults(3,:));
    end
end
toc





