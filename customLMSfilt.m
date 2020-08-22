classdef customLMSfilt < handle
   properties
      N {mustBeNumeric}
      M {mustBeNumeric}
      StepSize
      Coefficients
      refBuff
      %micBuff
      hank
   end
   methods
       function obj = customLMSfilt(N, M, mu)
           obj.N = N;
           obj.M = M;
           obj.StepSize = mu;
           obj.refBuff = dsp.AsyncBuffer(N+M-1); 
           write(obj.refBuff, zeros(N+M-1,1));
           obj.Coefficients = zeros(N,1);
           obj.hank = hankel(1:N,N:N+M-1);
       end
       
       function [yfilt, e, w] = step(obj, x, y)
           read(obj.refBuff, obj.M); % Drop the first M samples
           write(obj.refBuff, x);
           refBuffer = peek(obj.refBuff);
           refBufferMat = refBuffer(obj.hank);
           
           e = zeros(obj.M,1);
           yfilt = zeros(obj.M,1);
           
           for i = 1:obj.M               
               yfilt(i) = obj.Coefficients'*refBufferMat(:,i);
               e(i) = y(i) - yfilt(i);

               %LMS filter update
               %obj.Coefficients = obj.Coefficients + obj.StepSize*refBufferMat(:,i)*e(i);
               
               %Normalized LMS filter update
               alpha = 1/ (sum(sum(refBufferMat(:,i).^2))+0.000000001);
               obj.Coefficients = obj.Coefficients + obj.StepSize*alpha*refBufferMat(:,i)*e(i);
           end
           w = obj.Coefficients;
       end
       
       function [yfilt, e, w] = filtStep(obj, x, y)
            read(obj.refBuff, obj.M); % Drop the first M samples
            write(obj.refBuff, x);
            refBuffer = peek(obj.refBuff);
            yfilt = obj.Coefficients'*refBuffer(obj.hank);
            e = y - yfilt';
            yfilt = yfilt';
            w = obj.Coefficients;
            
       end
   end
end