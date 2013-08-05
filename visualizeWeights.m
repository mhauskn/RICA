function [] = visualizeWeights(theta, layersizes, data)
%VISUALIZEWEIGHTS Summary of this function goes here
%   Detailed explanation goes here
layersizes = [size(data,1) layersizes];
l = length(layersizes);
lnew = 0;
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W{i} = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
end
maxN = max(W{i}(:));
minN = min(W{i}(:));
normalized = (W{i} - minN)./(maxN-minN);
toShow = zeros(28, 28*layersizes(2));
for i=1:layersizes(2)
    toShow(:, (i-1)*28+1:(i-1)*28+28) = reshape(normalized(i,:), 28, 28);
end
imshow(toShow)

end

