clear all
close all
warning off 
clc


c = 100.00;
load SpeechData.mat T Y MinT MaxT

[N,R] = size(T);


T = T';
Y = Y';
N = length(Y);

SGM = [];  TOTAL = [];  Fbest = inf;

for s = 0.1:0.1:10;
    for i = 1:N;
        for j = 1:N;
            H(i,j) = Y(i)*Y(j)*GaussKernel(T(:,i),T(:,j),s);
        end
    end

    
    f = -ones(N,1);
    A = [];
    b = [];
    Aeq = Y ;
    beq = 0 ;
    vlb = zeros(N,1);
    vub = c*ones(N,1);
    x0 = zeros(N,1);
     opts = optimset('Display','off');
    
    [alfas] = quadprog(H,f,A,b,Aeq,beq,vlb,vub,x0,opts);
    SV = (find(abs(alfas)>0.00001));
    
    w = 0 ;
    for n = 1:N
        w = w + alfas(n)*Y(n).*T(:,n);
    end
    for i = 1:N
        xtest = T(:,i);
        y(i) = 0;
        for j=1:length(SV)
            y(i) = y(i)+alfas(SV(j))*Y(SV(j))*GaussKernel(T(:,SV(j)),xtest,s);
        end
    end
    
    
   NUMofERR = sum([Y~=sign(y)]);
   NUMofSVs = length(SV);
   SGM = [SGM;s];
   TOTAL = [TOTAL ; NUMofERR+NUMofSVs];
   
   if NUMofERR+NUMofSVs<Fbest
       Fbest = NUMofERR+NUMofSVs;
       ALPHASbest = alfas;
       SVbest = SV;
       SGbest = s;
   end
  
   disp(['sigma',num2str(s)])
   disp(['destek vektor sayisi:',num2str(length(SV))])
   disp(['hata sayisi:',num2str(NUMofERR)])
   
end

save SVMBEST.mat ALPHASbest SVbest SGbest T Y MinT MaxT

plot(SGM,TOTAL)
xlabel('\Sigma')
ylabel('NUMofERR+NUMofSVs')
set(gcf,'color',[1 1 1])
set(gcf,'Position',[348 42 804 500])
grid

    
   
    
    

