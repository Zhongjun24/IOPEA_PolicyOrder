function y=truecost(Q,c,h,b,mu,mean1,halfrange,A1,A2,demandtype,Sl,Sh)
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
O=zeros(samplesize,stablizationsize);
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
            O(index1,index2)=max(0,O(index1,index2-1)-d2(index1,index2-1)+Q);
        end
    end
    for index3=1:samplesize
        DminusO(index3)=d1(index3)-O(index3,stablizationsize);
    end
    accending=sort(DminusO);
    optimalS=min(Sh,max(Sl,accending(ceil(samplesize*(b-c)/(h+b-c)))));
    optimalhbcost=h*mean(max(0,optimalS-DminusO))+(b-c)*mean(max(0,DminusO-optimalS));
    if demandtype==1
        y=c*(mu-Q)+optimalhbcost;
    else if demandtype==2
            y=c*(mean1-Q)+optimalhbcost;
        else
            y=c*(A1*A2-Q)+optimalhbcost;
        end
    end
end