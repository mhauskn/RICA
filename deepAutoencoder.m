function [cost,grad] = deepAutoencoder(theta, layersizes, layerinds, data, top)
% TODO: This could be eliminated for speed
for i=1:length(layersizes)-1
    W{i} = reshape(theta(layerinds(i):layerinds(i+1)-1), layersizes(i+1), layersizes(i));
end

%% Forwards Prop
for i=1:length(layersizes)-1
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
end

%% COMPUTE COST
diff = h{i} - data;
M = size(data,2);

lambda = 0; % Lambda trades off between sparsity and reconstruction
s = log(cosh(h{top}));
cost = 1/M * (sum(diff(:).^2) + lambda * sum(s(:)));

% TODO: Compute full grad at once or go layer by layer?
grad = zeros(size(theta));
for i=1:length(layersizes)-1
  if i==1
    grad(layerinds(i):layerinds(i+1)-1) = 1/M * (2 * W{i+1}' * diff * data');
  else
    grad(layerinds(i):layerinds(i+1)-1) = 1/M * (2 * diff * h{1}');
  end
end


