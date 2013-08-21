function [cost,grad] = deepAutoencoder(layerWeights, layersizes, data, top, theta)
% cost and gradient of a deep autoencoder 
% layersizes is a vector of sizes of hidden layers, e.g., 
% layersizes[2] is the size of layer 2
% this does not count the visible layer
% data is the input data, each column is an example
% the activation function of the last layer is linear, the activation
% function of intermediate layers is the hyperbolic tangent function

% WARNING: the code is optimized for ease of implemtation and
% understanding, not speed nor space

%% FORCING THETA TO BE IN MATRIX FORMAT FOR EASE OF UNDERSTANDING
% Note that this is not optimized for space, one can just retrieve W and b
% on the fly during forward prop and backprop. But i do it here so that the
% readers can understand what's going on

% TODO: May need to add biases back in
l = length(layersizes);
lnew = 0;
for i=1:top
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    if i == top
        W{i} = reshape(layerWeights, layersizes(i+1), layersizes(i));
    else
        W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    end
end
% handle tied-weight stuff
j = 1;
for i=top+1:2*top
    W{i} = W{top + 1 - j}';
    j = j + 1;
end
%assert(lnew == length(theta), 'Error: dimensions of theta and layersizes do not match\n')

%% FORWARD PROP
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

grad = zeros(size(layerWeights));
if top == 1
    Wgrad = 1/M * (2 * W{top} * (dd + dd') + lambda * tanh(h{top}) * data');
else
    Wgrad = 1/M * (2 * W{top} * (dd + dd') + lambda * tanh(h{top}) * h{top-1}');    
end
grad = Wgrad(:);
end

