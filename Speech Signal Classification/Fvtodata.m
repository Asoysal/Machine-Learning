clear all
close all
clc

T = [];
Y = [];

for k = 1:60  %kaç tane denizli sesi varsa o kadar 
    FILENAME = ['KALIN00',num2str(k),'.wav'];
    [x] = audioread(FILENAME);
    [FV] = GetFeatureVector(x);
    T = [T;FV];
    Y = [Y;+1];
end

for k = 1:60  % kaç tane hatay sesi varsa o kadar
    FILENAME = ['INCE00',num2str(k),'.wav'];
    [x] = audioread(FILENAME);
    [FV] = GetFeatureVector(x);
    T = [T;FV];
    Y = [Y;-1];
end

MinT = min(T);
MaxT = max(T);

for i=1:size(T,2)
    T(:,i) = [T(:,i)-MinT(i)]/[MaxT(i)-MinT(i)];
end

save SpeechData.mat T Y MinT MaxT