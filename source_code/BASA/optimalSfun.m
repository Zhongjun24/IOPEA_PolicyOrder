function y=optimalSfun(optimalQ,Sl,Sh,c,b,h,demandtype,mu,mean1,halfrange,A1,A2)
stablizationsize=2000;
samplesize=2000;
if demandtype==1
    d1=exprnd(mu,1,samplesize);  %demand exponential distribution vector dimension(1,3T)
else if demandtype==2
        d1=2*halfrange*rand(1,samplesize)+mean1-halfrange; %demand uniform distribution vector dimension(1,3T)
    else
        d1=gamrnd(A1,A2,1,samplesize);  %demand gamma distribution vector dimension(1,3T)
    end
end
O1=zeros(samplesize,stablizationsize);
DminusO=zeros(1,samplesize);
accending=zeros(1,samplesize);
if demandtype==1
    d2=exprnd(mu,samplesize,stablizationsize);  %demand exponential distribution vector dimension(1,3T)
else if demandtype==2
        d2=2*halfrange*rand(samplesize,stablizationsize)+mean1-halfrange; %demand uniform distribution vector dimension(1,3T)
    else
        d2=gamrnd(A1,A2,samplesize,stablizationsize);  %demand gamma distribution vector dimension(1,3T)
    end
end
    for index1=1:samplesize
        for index2=2:stablizationsize
            O1(index1,index2)=max(0,O1(index1,index2-1)-d2(index1,index2-1)+optimalQ);
        end
    end
    for index3=1:samplesize
        DminusO(index3)=d1(index3)-O1(index3,stablizationsize);
    end
    accending=sort(DminusO);
    y=min(Sh,max(Sl,accending(ceil(samplesize*(b-c)/(h+b-c)))));
end