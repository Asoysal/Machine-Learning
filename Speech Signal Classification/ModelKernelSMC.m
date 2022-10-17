function [yhat]=ModelKernelSMC(test,lambdas,T,Y,s)
N=length(lambdas);
yhat=0;
for i=1:N
    yhat=yhat+lambdas(i)*Y(i)*GaussKernel(test,T(i,:)',s);
end

