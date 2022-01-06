function y=filter_weight(x, threshold, options)

K=length(x);
y=zeros(K,1);

if threshold > 0.5
   options="hard"; % HLNIROM
end

if options== "hard"
    for i=1:K
        if x(i)==max(x)
            y(i)=x(i);
        else
            y(i)=0;
        end
    end
end

if options == "soft_step"
    for i=1:K
        if x(i)>=threshold
            y(i)=x(i);
        else
            y(i)=0;
        end
    end
end

if options == "soft_decay"
    lambda = 4;
    a=(1-lambda)/threshold^lambda;
    b=lambda/threshold^(lambda-1);
    
    for i=1:length(y)
        if x(i)>=threshold
            y(i)=x(i);
        else
            y(i)=a*x(i)^(lambda+1)+b*x(i)^lambda;
        end
    end
end

y=y/sum(y);

end