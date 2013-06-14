%% Random initialization
% X. Glorot, Y. Bengio. 
% Understanding the difﬁculty of training deep feedforward neural networks.
% AISTATS 2010.
% QVL: this initialization method appears to perform better than 
% theta = randn(d,1);
s0 = size(traindata,1);
layersizes = [s0 layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
    A = rand(layersizes(i+1), layersizes(i))*2*r - r; %reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    theta(lold:lnew) = A(:);
end
theta = theta';
layersizes = layersizes(2:end);