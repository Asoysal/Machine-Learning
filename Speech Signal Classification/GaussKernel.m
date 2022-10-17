function[K]= GaussKernel(u,v,s)
K=exp(-(norm(u-v)^2)/(2*s^2));

%Burdaki s kernel parametresidir. Uygun s deðeri= s+destek vektör sayýsý (sv) en
%küçük olandýr.
%Hata + destek vektör sayýsý min olan deðer en iyi s deðeridir.
