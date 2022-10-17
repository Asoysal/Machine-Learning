clear all 
close all
clc




recObj = audiorecorder ;
disp('Start speaking.')
recordblocking(recObj, 5);
disp('End Of Recording.');

%Kaydý dinlemek için .. play(recObj);

x = getaudiodata(recObj);

%Çizdirmek için
plot(x);  

% dosyayý kaydetmek için
Fs = recObj.SampleRate;
audiowrite('HATAY010.wav',x,Fs)   


%Dosyadan okumak için
[y,Fs] = audioread('HATAY010.wav');


%Dinlemek için
sound(y,Fs)

a = GetFeatureVector(y) %her bir ses için elde ettiðin Feature vector olan 'a' yý SVM de o ses dosyasýna ait giriþ olarak kullanacaðýz. 
                        %buna karþý çýkýþta örneðin evet için +1 , hayýr
                        %için -1 þeklinde verilip sýnýflandýrma
                        %gerçekleþtirilecek.
                        %Bu her bir sese ait a yý excelde kaydet(her bir sese ait a inputlar olacak) buna uygun
                        %-1 ya da +1 olarak çýkýþ ver ve optimizasyondaki
                        %gibi data import et. En iyi sigmayý(s yi) bul.
                        %a için normalizasyon gerekebilir, içeriðine bak.







