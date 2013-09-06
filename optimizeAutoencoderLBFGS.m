function [] = optimizeAutoencoderLBFGS()
traindata = loadData('~/Desktop');
perm = randperm(size(traindata,2));
traindata = traindata(:,perm);

layersizes = [size(traindata,1) 100 20 10];

% Record the index that each layer starts at
indx = 1
for i=1:length(layersizes)-1
    layerinds(i) = indx;
    indx = indx + layersizes(i) * layersizes(i+1);
end
layerinds(length(layersizes)) = indx;

% Weight Initialization
% TODO: May need to add biases back in
for i=1:length(layersizes)-1
    r  = sqrt(6) / sqrt(layersizes(i+1)+layersizes(i));   
    A = rand(layersizes(i+1), layersizes(i))*2*r - r; 
    theta(layerinds(i):layerinds(i+1)-1) = A(:);
end
theta = theta';

addpath ~/Desktop/minFunc/
options.Method = 'lbfgs'; 
options.maxIter = 20;	  
options.display = 'on';
options.TolX = 1e-3;

batchSize = 1000;
maxIter = 20;
for layer=1:length(layersizes)-1
    fprintf('Training Layer %i\n',layer);
    for i=1:maxIter
        % Each iteration does a fresh batch looping when data runs out
        startIndex = mod((i-1) * batchSize, size(traindata,2)) + 1;
        fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
        data = traindata(:, startIndex:startIndex + batchSize-1);
        [theta, obj] = minFunc( @deepAutoencoder, theta, ...
            options, layersizes, layerinds, data, layer);        
    end
end

visualizeWeights(theta, layersizes, traindata)

