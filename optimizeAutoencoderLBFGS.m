function [] = optimizeAutoencoderLBFGS()
traindata = loadData('~/Desktop');
perm = randperm(size(traindata,2));
traindata = traindata(:,perm);

layersizes = [size(traindata,1) 100 50 20 10 size(traindata,1)];

% Record the index that each layer starts at
indx = 1;
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

% Found at http://www.di.ens.fr/~mschmidt/Software/minFunc.html
addpath(genpath('~/Desktop/minFunc_2012/'));
addpath(genpath('vis/'));

options.Method = 'lbfgs'; 
options.maxIter = 20;	  
options.display = 'on';
options.TolX = 1e-3;

passes = 0; % Number of passes over the dataset
batchSize = 1000;
for i=1:size(traindata,2)*passes/batchSize
    % Each iteration does a fresh batch looping when data runs out
    startIndex = mod((i-1) * batchSize, size(traindata,2)) + 1;
    fprintf('startIndex = %d, endIndex = %d\n', startIndex, startIndex + batchSize-1);
    data = traindata(:, startIndex:startIndex + batchSize-1);

    %% Optionally Check the Gradient
%     fastDerivativeCheck(@deepAutoencoder, theta, 1, 2, layersizes, layerinds, data, 1);
%     exit;

    [theta, obj] = minFunc(@deepAutoencoder, theta, options, ...
                           layersizes, layerinds, data);
end

visualizeWeights(theta, layersizes, layerinds, traindata)
visualizeMaximallyResponsiveInputs(theta, layersizes, layerinds, traindata)

exit