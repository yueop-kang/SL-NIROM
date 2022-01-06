function y=softmax(x)
c=-max(x);
y=exp(x+c)./sum(exp(x+c)')';

end