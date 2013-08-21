function [] = visualizeWeights(theta, layersizes, data)
%% VISUALIZEWEIGHTS This function shows the weights for each hidden neuron
l = length(layersizes);
lnew = 0;
transform = eye(layersizes(1));
for i=1:l-1
    lold = lnew + 1;
    lnew = lnew + layersizes(i) * layersizes(i+1);
    W = reshape(theta(lold:lnew), layersizes(i+1), layersizes(i));
    
    % Transform the weights back into the input space
    transform = W * transform;
    
    % Normalize the weights to [0,1]
    maxN = max(transform(:));
    minN = min(transform(:));
    normalized = 1-(transform - minN)./(maxN-minN);
    
    toShow = zeros(28, 28*layersizes(i+1));
    for j=0:layersizes(i+1)-1
        toShow(:, j*28+1:j*28+28) = reshape(normalized(j+1,:), 28, 28);
    end
    filename = strcat('images/layer', num2str(i), 'optimal.png');
    imwrite(toShow,filename)
end

end

