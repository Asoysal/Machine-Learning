clear all
close all
warning off
clc

load SVMBEST.mat ALPHASbest SVbest SGbest T Y MinT MaxT

recObj=audiorecorder;
disp('start speaking');
recordblocking(recObj,5);
disp('end of recording');
x=getaudiodata(recObj);
plot(x)
[FV]=GetFeatureVector(x)';
xtest=[(FV'-MinT)./(MaxT-MinT)]';
yhat=0;
for i=1:length(SVbest)
    yhat=yhat+ALPHASbest(SVbest(i))*Y(SVbest(i))*GaussKernel(T(:,SVbest(i)),xtest,SGbest);
end
yhat = sign(yhat);

if yhat>0
    ANS=('kalin')
else
     ANS=('ince')
end
h=text(length(x),max(x),ANS);
set(h,'FontSize',32);
set(h,'FontWeight','Bold');
set(h,'Color',[1 0 0]);
set(h,'EdgeColor',[0 0 1]);
set(h,'LineWidth',3);

set(gcf,'color',[1 1 1]);
set(gcf,'Position',[275 195 721 497]);
