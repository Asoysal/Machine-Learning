function[K]= GaussKernel(u,v,s)
K=exp(-(norm(u-v)^2)/(2*s^2));

%Burdaki s kernel parametresidir. Uygun s de�eri= s+destek vekt�r say�s� (sv) en
%k���k oland�r.
%Hata + destek vekt�r say�s� min olan de�er en iyi s de�eridir.
