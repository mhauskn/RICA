function [cost,grad] = deepAutoencoder(theta, layersizes, layerinds, data)
% TODO: This could be eliminated for speed
for i=1:length(layersizes)-1
    W{i} = reshape(theta(layerinds(i):layerinds(i+1)-1), layersizes(i+1), layersizes(i));
end

sparsityCost = 0;

%% Forwards Prop
for i=1:length(layersizes)-1
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
    s = log(cosh(h{i}));
    sparsityCost = sparsityCost + sum(s(:));
end

%% COMPUTE COST
diff = h{i} - data;
M = size(data,2);

lambda = 0.1; % Lambda trades off between sparsity and reconstruction
cost = 1/M * (sum(diff(:).^2) + lambda * sparsityCost);

% TODO: Compute full grad at once or go layer by layer?
grad = zeros(size(theta));
above = diff;
for i=length(layersizes)-1:-1:1
    if i < length(layersizes)-1
        above = W{i+1}' * above;
    end
    if i==1
        below = data';
    else
        below = h{i-1}';
    end
    grad(layerinds(i):layerinds(i+1)-1) = 1/M * (2 * above * below) + lambda * tanh(h{i} * below);
end
