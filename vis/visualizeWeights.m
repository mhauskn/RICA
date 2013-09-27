function [] = visualizeWeights(theta, layersizes, layerinds, data)
%% VISUALIZEWEIGHTS This function shows the weights for each hidden neuron
width = 28;
height = 28;
lnew = 0;
numLayers = length(layersizes)-1;
maxLayerSize = max(layersizes) / width;
toShow = zeros((height+1)*numLayers, (width+1)*maxLayerSize);

transform = eye(layersizes(1));
for i=1:numLayers
    W = reshape(theta(layerinds(i):layerinds(i+1)-1), layersizes(i+1), layersizes(i));
    
    % Transform the weights back into the input space
    transform = W * transform;
    
    % Normalize the weights to [0,1]
    maxN = max(transform(:));
    minN = min(transform(:));
    normalized = 1-(transform - minN)./(maxN-minN);
    
    % Write each row of the transform 
    ymin = (numLayers-i)*height+1+numLayers-i;
    ymax = (numLayers-i+1)*height+numLayers-i;
    for j=0:layersizes(i+1)-1
        xmin = j*width+1+j;
        xmax = (j+1)*width+j;
        toShow(ymin:ymax, xmin:xmax) = reshape(normalized(j+1,:), width, height);
    end
end
filename = strcat('images/weights.png');
imwrite(toShow,filename);
end


function [] = visualizeMaximallyResponsiveInputs(theta, layersizes, data)
%VISUALIZEMAXIMALLYRESPONSIVEINPUTS Shows the n inputs that create maximal
%responses from output neuron n
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

n = 25;
for i=0:layersizes(2)-1
    [~, sortIndex] = sort(h{1}(i+1,:),'descend');
    s = zeros(28*sqrt(n));
    for j=0:n-1
        col = mod(j,sqrt(n));
        row = floor(j/sqrt(n));
        s(row*28+1:row*28+28,col*28+1:col*28+28) = reshape(data(:,sortIndex(j+1)),28,28);
    end
    filename = strcat('images/n',num2str(i+1),'MaxResp.png');
    imwrite(s,filename);
end

end

