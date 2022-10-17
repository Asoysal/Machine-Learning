function [FV]=GetFeatureVector(x)

Fs = 8000;
Ts = 1/Fs;
N = length(x);
[X] = fft(x);

Magnitude = 2*abs(X)/N;
Magnitude(1) = Magnitude(1)/2;
Magnitude = Magnitude(N/2:end);
Magnitude = [Magnitude-min(Magnitude)]/[max(Magnitude)-min(Magnitude)];
F = [0:N/2]*Fs/N;
[V,I] = sort(Magnitude,'descend');
Max_F = I(1:5)*Ts;
FV = [Max_F'];

MIN = min(x);
FV = [FV,MIN];

MAX = max(x);
FV = [FV,MAX];

MEAN = mean(x);
FV = [FV,MEAN];

STD = std(x);
FV = [FV,STD];

ENERGY = sum(abs(x)*12);
FV = [FV,ENERGY];

% ENTROPY = wentropy(x,'shannon');
% FV = [FV,ENTROPY];

STD = std(x);
FV = [FV,STD];

Magnitude = 2*abs(X)/N;
Magnitude(1) = Magnitude(1)/2;
Magnitude = Magnitude(N/2:end);
Magnitude = [Magnitude-min(Magnitude)]/[max(Magnitude)-min(Magnitude)];
[yupper,ylower] = envelope(Magnitude,30,'peak');
[pks,locs]=findpeaks(yupper);
[V,I]=sort(pks,'Descend');
F = [0:N/2]*Fs/N;
IMSF = F(locs(I(I:5)))';
FV = [FV,IMSF'];

p=3;
[a]=lpc(x,p);
FV = [FV,[a(2:end)]];


% [coeffs,delta,deltaDelta,loc] = mfcc(x,Fs);
% FV = [FV,coeffs];
% FV = [FV,delta];
% FV = [FV,deltaDelta];


% [coeffs,delta,deltaDelta,loc] = mfcc(audioIn,fs,'LogEnergy','Replace','DeltaWindowLength',5)


% x = x/ max(max(x));
% window_length = 2 * Fs;
% step = 1 * Fs; 
% frame_num = floor((length(x)-window_length)/step) + 1;
% 
% zcr_ = zeros(frame_num, 1);
% wav_window2 = zeros(window_length, 1);
% pos = 1;
% 
% for i=1:frame_num
%    wav_window = x(pos:pos + window_length-1);
%    wav_window2(2:end) = wav_window(1:end-1);
%    zcr_(i) = 1/2 * sum(abs(sign(wav_window) - sign(wav_window2))) * Fs / window_length;
%    pos = pos + step;
% end
% 
% FV = [FV,zcr_'];

mf=[];
coeffs = mfcc(x,Fs);
 for i = 1:1:498
             if coeffs(i,1) ~= -inf 
                 mf = [mf,coeffs(i,:)];
             end
 end

for i = 1:1:400
FV = [FV,mf(1,i)];
end






