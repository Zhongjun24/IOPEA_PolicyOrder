function totalclairvoyantcost = main3(t_spec)

maxround=1; maxT=t_spec;
%Sl=0.1; Sh=4; Ql=0.1; Qh=1; h=1; b=[10]; c=0.5;
Sl=1; Sh=40; Ql=1; Qh=5; h=1; b=[10]; c=0.5;%for large scale
lR=5; %lead time for the regular source
demandtype=4; %demandtype 1 is exponential, 2 is uniform, 3 is gamma, 4 is normal
mu=1; %exponential demand mean
mean1=40/3;
std1 = 20/3;%uniform demand mean
halfrange=20; 
halfrange=min(halfrange,mean1); %uniform demand halfrange
A1=4; %gamma shape parameters
A2=1; %gamma scale parameter
gamma_coefficient=1; I_coefficient=1; L_coefficient=0.02; eta_coefficient=0.4;
arrayT=[t_spec];
maxhorizon=length(arrayT);

function d = clip_and_mask_demand(d, min_val, max_val, ratio_)
    d = min(max(d, min_val), max_val);
    mask = rand(size(d)) < ratio_;
    d(mask) = 0;
end

regretperround=zeros(maxround,maxhorizon);
maxm=ceil(0.5*log2(2*maxT/(L_coefficient*I_coefficient*(log(maxT))^2*(log(log(maxT)))^3)));
gamma=zeros(1,maxm);
I=zeros(1,maxm);
eta=zeros(1,6*maxT);
tau=zeros(maxT,maxm,3);
barQ=zeros(1,6*maxT);
barS=zeros(1,6*maxT);
S=zeros(maxm,3,maxT);
qR=zeros(1,6*maxT);
qE=zeros(1,6*maxT);
x=zeros(1,6*maxT+1);
o=zeros(1,6*maxT);
K=zeros(maxm,3,maxT);
Delta=zeros(maxm,3,maxT);
Gcurrentm=zeros(maxT,maxm,3);
totalG=zeros(maxT,maxm,3);
totalI=zeros(maxT,maxm,3);
G=zeros(maxT,maxm,3);
UB=zeros(maxT,maxm,3);
LB=zeros(maxT,maxm,3);
cost=zeros(1,6*maxT);
barQ_clairvoyant=zeros(1,maxT);
barS_clairvoyant=zeros(1,maxT);
qR_clairvoyant=zeros(1,maxT);
qE_clairvoyant=zeros(1,maxT);
x_clairvoyant=zeros(1,maxT+1);
o_clairvoyant=zeros(1,maxT);
cost_clairvoyant=zeros(1,maxT);
d=zeros(1,6*maxT);
v=zeros(maxT,maxm);
Q=zeros(maxT,3);
l=ones(1,maxT)*Ql;
r=ones(1,maxT)*Qh;
w=ones(1,maxT);
regret=zeros(length(b),length(arrayT),length(c));
stdregret=zeros(length(b),length(arrayT),length(c));
margin_error=zeros(length(b),length(arrayT),length(c));
z=1.96; %95% confidence interval value
for regretindexc=1:length(c)
    for regretindexb=1:length(b)
%*****below is to solve the claivoyant problem*****%
[optimalQ,optimaltruecost] = fminbnd(@(Q)truecost(Q,c(regretindexc),h,b(regretindexb),mu,mean1,halfrange,A1,A2,demandtype,Sl,Sh),Ql,Qh);
optimalS=optimalSfun(optimalQ,Sl,Sh,c(regretindexc),b(regretindexb),h,demandtype,mu,mean1,halfrange,A1,A2);
%*****above is to solve the claivoyant problem*****%
for horizonindex=1:maxhorizon
    T=arrayT(horizonindex);
    L=ceil(L_coefficient*log(T)*(log(log(T)))^2);   
for round=1:maxround
    %currentS=(Sl+Sh)/2;
    currentS=Qh;
if demandtype==1
    d=exprnd(mu,1,6*maxT);  %demand exponential distribution vector dimension(1,3T)
elseif demandtype==2
    d=2*halfrange*rand(1,6*maxT)+mean1-halfrange; %demand uniform distribution vector dimension(1,3T)
elseif demandtype == 3
    d=gamrnd(A1,A2,1,6*maxT);  %demand gamma distribution vector dimension(1,3T)
else
    d = std1 * randn(1, 6*maxT) + mean1;
end

d = clip_and_mask_demand(d, 0, 40, 0.3);

v(1,1)=0;
if horizonindex==maxhorizon && round==maxround
    v=zeros(maxT,maxm);
    Q=zeros(maxT,3);
    l=ones(1,maxT)*Ql;
    r=ones(1,maxT)*Qh;
    w=ones(1,maxT);
end
for n=1:1:T
    S=zeros(maxm,3,maxT);
    Delta=zeros(maxm,3,maxT);
    K=zeros(maxm,3,maxT);
    if v(n,1)<T
    w(n)=r(n)-l(n);
    Q(n,1)=l(n)+w(n)/4;
    Q(n,2)=l(n)+w(n)/2;
    Q(n,3)=l(n)+3*w(n)/4;
    for m=1:1:maxm
        if v(n,m)<T
        gamma(m)=gamma_coefficient*(2^(-m));
        I(m)=ceil(I_coefficient*log(T)*log(log(T))/(2^(-2*m)));
        for j=1:1:3
            tau(n,m,j)=v(n,m)+(j-1)*I(m)*L;
            S(m,j,1)=currentS;
            for i=1:1:I(m)
                %eta(i)=eta_coefficient*(Sh-Sl)/(max(h,b-c)*sqrt(i));
                eta(i)=eta_coefficient/sqrt(i);
                for t=(tau(n,m,j)+(i-1)*L+1):1:(tau(n,m,j)+i*L)
                    barQ(t)=Q(n,j);
                    barS(t)=S(m,j,i);
                    qR(t)=Q(n,j);
                    if t>=lR+1
                        qE(t)=max(0,S(m,j,i)-x(t)-qR(t-lR));
                        o(t)=max(0,x(t)+qR(t-lR)-S(m,j,i));
                        %cost(t)=c*qE(t)+h*max(0,S(m,j,i)+o(t)-d(t))+b*max(0,d(t)-S(m,j,i)-o(t));
                        cost(t)=b(regretindexb)*d(t)-c(regretindexc)*Q(n,j)+h*max(0,S(m,j,i)+o(t)-d(t))-(b(regretindexb)-c(regretindexc))*min(S(m,j,i)+o(t),d(t));
                    else
                        qE(t)=max(0,S(m,j,i)-x(t));
                        o(t)=max(0,x(t)-S(m,j,i));
                        %cost(t)=c*qE(t)+h*max(0,S(m,j,i)+o(t)-d(t))+b*max(0,d(t)-S(m,j,i)-o(t));
                        cost(t)=b(regretindexb)*d(t)-c(regretindexc)*Q(n,j)+h*max(0,S(m,j,i)+o(t)-d(t))-(b(regretindexb)-c(regretindexc))*min(S(m,j,i)+o(t),d(t));
                    end
                    x(t+1)=max(0,o(t)+S(m,j,i)-d(t));
                end
                K(m,j,i)=-c(regretindexc)*Q(n,j)+h*max(0,o(tau(n,m,j)+i*L)+S(m,j,i)-d(tau(n,m,j)+i*L))...
                           -(b(regretindexb)-c(regretindexc))*min(o(tau(n,m,j)+i*L)+S(m,j,i),d(tau(n,m,j)+i*L));
                if o(tau(n,m,j)+i*L)+S(m,j,i)-d(tau(n,m,j)+i*L)>=0
                    Delta(m,j,i)=h;
                else
                    Delta(m,j,i)=c(regretindexc)-b(regretindexb);
                end
                S(m,j,i+1)=max(Sl,min(Sh,S(m,j,i)-eta(i)*Delta(m,j,i)));
                currentS=S(m,j,i+1);
            end
            %G(n,m,j)=(1/I(m))*sum(K(n,m,j,1:1:I(m)));
            Gcurrentm(n,m,j)=sum(K(m,j,1:1:I(m)));
            if m==1
                totalG(n,m,j)=Gcurrentm(n,m,j);
                totalI(n,m,j)=I(m);
            else 
                totalG(n,m,j)=totalG(n,m-1,j)+Gcurrentm(n,m,j);
                totalI(n,m,j)=totalI(n,m-1,j)+I(m);
            end
            G(n,m,j)=totalG(n,m,j)/totalI(n,m,j);
            UB(n,m,j)=G(n,m,j)+gamma(m)/2;
            LB(n,m,j)=G(n,m,j)-gamma(m)/2;
        end
        if max(LB(n,m,1),LB(n,m,3))>=min(UB(n,m,1),UB(n,m,3))+gamma(m)
            if LB(n,m,1)>=LB(n,m,3)
                l(n+1)=Q(n,1); r(n+1)=r(n);
            else
                l(n+1)=l(n); r(n+1)=Q(n,3);
            end
            v(n+1,1)=tau(n,m,3)+I(m)*L;
            break
        else if max(LB(n,m,1),LB(n,m,3))>=UB(n,m,2)+gamma(m)
                if LB(n,m,1)>=LB(n,m,3)
                    l(n+1)=Q(n,1); r(n+1)=r(n);
                else
                    l(n+1)=l(n); r(n+1)=Q(n,3);
                end
                v(n+1,1)=tau(n,m,3)+I(m)*L;
                break
            else if m<maxm
                    v(n,m+1)=tau(n,m,3)+I(m)*L;
                else
                    v(n+1,1)=tau(n,m,3)+I(m)*L;
                    l(n+1)=l(n); r(n+1)=r(n);
                end
            end
        end 
        else
            v(n+1,1)=v(n,m);
            break
        end
    end
    else
        break
    end
end
totalcost=sum(cost(1:T))+c(regretindexc)*x(T+1);
optimalQ = (l(T) + r(T))/2;
optimalS = currentS;
%*****below is to apply the claivoyant problem*****%      
for t=1:T
    barQ_clairvoyant(t)=optimalQ;
    barS_clairvoyant(t)=optimalS;
    qR_clairvoyant(t)=optimalQ;
    if t>=lR+1
        qE_clairvoyant(t)=max(0,optimalS-x_clairvoyant(t)-qR_clairvoyant(t-lR));
        o_clairvoyant(t)=max(0,x_clairvoyant(t)+qR_clairvoyant(t-lR)-optimalS);
        %cost_clairvoyant(t)=c*qE_clairvoyant(t)+h*max(0,optimalS+o_clairvoyant(t)-d(t))+b*max(0,d(t)-optimalS-o_clairvoyant(t));
        cost_clairvoyant(t)=b(regretindexb)*d(t)-c(regretindexc)*optimalQ+h*max(0,optimalS+o_clairvoyant(t)-d(t))-(b(regretindexb)-c(regretindexc))*min(optimalS+o_clairvoyant(t),d(t));
    else
        qE_clairvoyant(t)=max(0,optimalS-x_clairvoyant(t));
        o_clairvoyant(t)=max(0,x_clairvoyant(t)-optimalS);
        %cost_clairvoyant(t)=c*qE_clairvoyant(t)+h*max(0,optimalS+o_clairvoyant(t)-d(t))+b*max(0,d(t)-optimalS-o_clairvoyant(t));
        cost_clairvoyant(t)=b(regretindexb)*d(t)-c(regretindexc)*optimalQ+h*max(0,optimalS+o_clairvoyant(t)-d(t))-(b(regretindexb)-c(regretindexc))*min(optimalS+o_clairvoyant(t),d(t));
    end
    x_clairvoyant(t+1)=max(0,o_clairvoyant(t)+optimalS-d(t));
end
totalclairvoyantcost=sum(cost_clairvoyant(1:T))+c(regretindexc)*x_clairvoyant(T+1);
%*****below is to compute the regret of each round each T*****%
regretperround(round,horizonindex)=(totalcost-totalclairvoyantcost)/totalclairvoyantcost;
end
end
regret(regretindexb,:,regretindexc)=mean(regretperround);
stdregret(regretindexb,:,regretindexc)=std(regretperround);
margin_error(regretindexb,:,regretindexc)=z/sqrt(maxround)*stdregret(regretindexb,:,regretindexc);
    end
end
totalclairvoyantcost;
end