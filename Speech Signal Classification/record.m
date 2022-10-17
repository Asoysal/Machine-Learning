clear all 
close all
clc




recObj = audiorecorder ;
disp('Start speaking.')
recordblocking(recObj, 5);
disp('End Of Recording.');

%Kayd� dinlemek i�in .. play(recObj);

x = getaudiodata(recObj);

%�izdirmek i�in
plot(x);  

% dosyay� kaydetmek i�in
Fs = recObj.SampleRate;
audiowrite('HATAY010.wav',x,Fs)   


%Dosyadan okumak i�in
[y,Fs] = audioread('HATAY010.wav');


%Dinlemek i�in
sound(y,Fs)

a = GetFeatureVector(y) %her bir ses i�in elde etti�in Feature vector olan 'a' y� SVM de o ses dosyas�na ait giri� olarak kullanaca��z. 
                        %buna kar�� ��k��ta �rne�in evet i�in +1 , hay�r
                        %i�in -1 �eklinde verilip s�n�fland�rma
                        %ger�ekle�tirilecek.
                        %Bu her bir sese ait a y� excelde kaydet(her bir sese ait a inputlar olacak) buna uygun
                        %-1 ya da +1 olarak ��k�� ver ve optimizasyondaki
                        %gibi data import et. En iyi sigmay�(s yi) bul.
                        %a i�in normalizasyon gerekebilir, i�eri�ine bak.







