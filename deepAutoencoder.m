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
    j++;
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
cost = 1/M * 0.5 * sum(diff(:).^2); 
% cost = sum(diff(:).^2); % TODO: This is the cost func i used

assert(l == 2)
grad = zeros(size(theta));
for i=1:l-1
    Wgrad{i} = 2 * W{i} * (data * diff' + diff * data');
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    grad(lold:lnew) = Wgrad{i}(:);
end

end
