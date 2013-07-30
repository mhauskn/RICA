function [cost,grad] = deepAutoencoder(theta, layersizes, data)
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
layersizes = [size(data,1) layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
end
% handle tied-weight stuff
j = 1;
for i=l:2*(l-1)
    W{i} = W{l - j}';
    j = j + 1;
end
assert(lnew == length(theta), 'Error: dimensions of theta and layersizes do not match\n')

%% FORWARD PROP
for i=1:2*(l-1)
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
end

%% COMPUTE COST
diff = h{i} - data; 
M = size(data,2);
lambda = .1; % Lambda trades off between sparsity and reconstruction
s = log(cosh(h{1}));
cost = 1/M * (sum(diff(:).^2) + lambda * sum(s(:)));

assert(l == 2)
lnew = 0;
grad = zeros(size(theta));
dd = data * diff';
for i=1:l-1
    Wgrad{i} = 1/M * (2 * W{i} * (dd + dd') + lambda * tanh(h{1}) * data');
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    grad(lold:lnew) = Wgrad{i}(:);
end
end

