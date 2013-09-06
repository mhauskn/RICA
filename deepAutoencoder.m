function [cost,grad] = deepAutoencoder(theta, layersizes, layerinds, data, top)
for i=1:top
    W{i} = reshape(theta(layerinds(i):layerinds(i+1)-1), layersizes(i+1), layersizes(i));
end
% handle tied-weight stuff
j = 1;
for i=top+1:2*top
    W{i} = W{top + 1 - j}';
    j = j + 1;
end

%% Forwards & Backwards Prop
for i=1:2*top
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
end

%% COMPUTE COST
if top == 1
    diff = h{i} - data;
    M = size(data,2);
    dd = data * diff';
else
    diff = h{i-(top-1)} - h{top-1};
    M = size(h{top-1},2);
    dd = h{top-1} * diff';
end

lambda = .1; % Lambda trades off between sparsity and reconstruction
s = log(cosh(h{top}));
cost = 1/M * (sum(diff(:).^2) + lambda * sum(s(:)));

grad = zeros(size(theta));
if top == 1
    Wgrad = 1/M * (2 * W{top} * (dd + dd') + lambda * tanh(h{top}) * data');
else
    Wgrad = 1/M * (2 * W{top} * (dd + dd') + lambda * tanh(h{top}) * h{top-1}');    
end
lnew = 0;
for i=1:length(layersizes)-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    if i == top
        grad(lold:lnew) = Wgrad(:);
    end
end
end

