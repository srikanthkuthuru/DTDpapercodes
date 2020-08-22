function [] = openLoopExps() 
%%

sd = "/home/srikanth/Documents/sednn-master/mixture2clean_dnn/full_data/train_speech/";
speechDir = dir(sd);
nd = "/home/srikanth/Documents/sednn-master/mixture2clean_dnn/full_data/train_noise/";
noiseDir = dir(nd);


nSegs = 10;
noiseType = 0;

N = 512; 
M = 160;
nccLen = 3*M;
pK = 5; %Buffer length for tracking peak delays
fs = 16000;
hank = hankel(1:N,N:N+nccLen-1);
%regu = gpuArray(0.0000001*eye(N)); %regularizer
regu = 0.000000001*eye(N); %regularizer

T = 0.7;
Tol = 0.7;
Tncc = 0.9993;
VAD_cst_param = vadInitCstParams;

ntrials = 50;
SNRall = [100];
NFRall = [-10,-5,0,5,10,15,20];

RecallResults = zeros([3,length(SNRall),length(NFRall)]);
PrecisionResults = zeros([3,length(SNRall),length(NFRall)]);

for SNR_ind = 1:length(SNRall)
    for NFR_ind = 1:length(NFRall)
        RtempResults = zeros([3,ntrials]);
        PtempResults = zeros([3,ntrials]);
        parfor trial = 1:ntrials
            SNR = SNRall(SNR_ind);
            NFR = NFRall(NFR_ind);

            h = hsample(N);
            %[~,truepos] = max(h);
            %plot(h);
            [mixedOut,refOut,nearOut] = DTDdataSampler(speechDir, noiseDir, h, nSegs, NFR, SNR, noiseType);
            %figure(1); subplot(311);plot(refOut);subplot(312);plot(nearOut);subplot(313);plot(mixedOut);
            %sound(mixedOut, fs);
            %waitforbuttonpress;

            %
            SpkrSrc   = dsp.SignalSource('Signal',refOut,'SamplesPerFrame',M);
            micSrc    = dsp.SignalSource('Signal',mixedOut,'SamplesPerFrame',M);
            nearSrc    = dsp.SignalSource('Signal',nearOut,'SamplesPerFrame',M);

            xbuff = dsp.AsyncBuffer(N+nccLen); write(xbuff, zeros(N+nccLen,1));
            ybuff = dsp.AsyncBuffer(nccLen); write(ybuff, zeros(nccLen,1));
            pbuff = dsp.AsyncBuffer(pK);write(pbuff, zeros(pK,1));

            dtdValArr1 = [];
            dtdValArr2 = [];
            dtdValArr3 = [];
            refVAD = []; 
            nearVAD = [];


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



            ref = []; near = []; mic = []; echoref = [];
            ref.vadState.biquad = [];
            near.vadState.biquad = [];
            mic.vadState.biquad = [];
            for i = 1:10
                [ref.vadStatus, ref.vadState] = vad729custom(0.01*randn(80,1),ref.vadState, VAD_cst_param);
                [near.vadStatus, near.vadState] = vad729custom(0.0001*randn(80,1),near.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(0.0001*randn(80,1),mic.vadState, VAD_cst_param);
            end

            while(~isDone(micSrc))

                echoref.frame = SpkrSrc();
                mic.frame = micSrc();
                near.frame = nearSrc();

                % Echoref Buffer Update 
                read(xbuff, M); %Drop the first M samples
                write(xbuff,echoref.frame);
                echoref.buffer = peek(xbuff);

                % Mic Buffer Update
                read(ybuff, M); % Drop the first M samples
                write(ybuff,mic.frame);
                mic.buffer = peek(ybuff); % Get entire buffer 

                %VAD trackers
                [ref.vadStatus, ref.vadState] = vad729custom(downsample(echoref.frame,2),ref.vadState, VAD_cst_param);
                [near.vadStatus, near.vadState] = vad729custom(downsample(near.frame,2),near.vadState, VAD_cst_param);
                [mic.vadStatus, mic.vadState] = vad729custom(downsample(mic.frame,2),mic.vadState, VAD_cst_param);

                micbuff = mic.buffer(end-nccLen+1:end);
                echorefbuff = echoref.buffer;

                if(mic.vadStatus == 0 || ref.vadStatus == 0)
                    % No doubletalk
                    c1 = 1;
                    c4 = 1;
                    c3 = 1;
                else
                    %tic
                    %-------------------------------------------------------------------------------------%
                    %Common terms across methods
                    xmul = echorefbuff(hank);
                    Rxy = (1/nccLen)*abs(xmul*micbuff);
                    Rxx = (1/nccLen)*(xmul*xmul') + regu;
                    RxxInv = inv(Rxx);

                    sigy = sqrt(mean(micbuff.^2));


                    %------- Open loop version - normal cross correlation
                    den = sqrt( mean(echorefbuff.^2))*sigy;
                    cvec = flipud(Rxy/den);
                    [c1, ~] = max(cvec);

                    %--------- Normalized cross corr
                    %c2 = (1/sigy)*sqrt(Rxy'* (RxxInv*Rxy));
                    %fprintf('c2 = %d \n', c2);

                    %--------- Peak Tracking DTD - modified with Rxx-1 (Cholesky Whitening)
                    xmulNorm = RxxInv*xmul; %Need smoothed Rxx here!!
                    RxyNew = (1/nccLen)*abs(xmulNorm*micbuff);
                    cvec = flipud(RxyNew/sigy);
                    [~, mpos] = max(cvec);
                    %fprintf('mpos = %d,.....truepos = %d \n', mpos, truepos);
                    %figure(2);plot(cvec);

                    % Peak echo Buffer Update 
                    read(pbuff, 1); %Drop the first sample
                    write(pbuff,mpos);
                    c3 = (std(peek(pbuff)) == 0);

                    %-------NCC with Expectation vals -- works better than prev one
                    xmulNorm = sqrtm(RxxInv)*xmul; %Need smoothed Rxx here!!
                    RxyNew = (1/nccLen)*abs(xmulNorm*micbuff);
                    cvec = flipud(RxyNew/sigy);
                    c4 = norm(cvec,2);
                    %fprintf('...................c4 = %d \n', c4);

                    %toc
                    %------------------------------------------------------------------------------------% 
                end

                dtdValArr1 = [dtdValArr1; c1*ones(M,1)]; 
                dtdValArr2 = [dtdValArr2; c4*ones(M,1)];
                dtdValArr3 = [dtdValArr3; c3*ones(M,1)];


                refVAD = [refVAD; ref.vadStatus*ones(M,1)];
                nearVAD = [nearVAD; near.vadStatus*ones(M,1)];


                %{
                AECScope1([echoref.frame],...
                    [near.frame],...
                    [mic.frame],...
                    (c1>Tol)*(ones(M,1)),...
                    (c4>Tncc)*(ones(M,1)),...
                    c3*(ones(M,1)));
                %}


            end
            dtdTrue = (refVAD>T).*(nearVAD>T);
            dtdValArr1 = 1 - (dtdValArr1 > Tol);
            dtdValArr2 = 1 - (dtdValArr2 > Tncc);
            dtdValArr3 = 1 - dtdValArr3;
            
            RtempResults(:, trial) = [sum(and(dtdValArr1,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr2,dtdTrue))/sum(dtdTrue), sum(and(dtdValArr3,dtdTrue))/sum(dtdTrue)];
            fprintf('\n SNR = %d, NFR = %d, trial = %d, Recalls = %d, %d, %d \n', SNR, NFR, trial, RtempResults(:, trial));
            
            PtempResults(:, trial) = [sum(and(dtdValArr1,dtdTrue))/sum(dtdValArr1), sum(and(dtdValArr2,dtdTrue))/sum(dtdValArr2), sum(and(dtdValArr3,dtdTrue))/sum(dtdValArr3)];
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


%{
figure(1);hold on;
plot(NFRall, squeeze(RecallResults(1,1,:))', '-o')
plot(NFRall, squeeze(RecallResults(2,1,:))', '-s')
plot(NFRall, squeeze(RecallResults(3,1,:))', '-^')
xlabel('NFeR (dB)'); ylabel('Recall Score');
ylim([0,1]);
legend('Open Loop Cross Correlation', 'NCC', 'Peak Tracking DTD with P=1', 'Location', 'southeast')


figure(1);hold on;
plot(NFRall, squeeze(PrecisionResults(1,1,:))', '-o')
plot(NFRall, squeeze(PrecisionResults(2,1,:))', '-s')
plot(NFRall, squeeze(PrecisionResults(3,1,:))', '-^')
xlabel('NFeR (dB)'); ylabel('Precision Score');
ylim([0,1]);
legend('Closed Loop Cross Correlation', 'Closed loop NCC', 'FastPT with P=5', 'Location', 'southeast')


%}
end














