function [] = visualizeMaximallyResponsiveInputs(theta, layersizes, layerinds, data)
%VISUALIZEMAXIMALLYRESPONSIVEINPUTS Shows the n inputs that create maximal
%responses from output neuron n
nLayers = length(layersizes)-1;
for i=1:nLayers
    W{i} = reshape(theta(layerinds(i):layerinds(i+1)-1), layersizes(i+1), layersizes(i));
end

%% Forwards Prop
for i=1:nLayers
    if i==1
        h{i} = W{i} * data;
    else
        h{i} = W{i} * h{i-1};
    end
end

%% Make a 5x5 grid of the 25 maximally responsive units
nExamples = 25;
for l=1:nLayers
    for i=1:layersizes(l+1)
        %% Sort the activations of all the data samples for the hidden unit h{l}(i)
        [~, sortIndex] = sort(h{l}(i,:),'descend');
        s = zeros(28*sqrt(nExamples));
        for j=0:nExamples-1
            col = mod(j,sqrt(nExamples));
            row = floor(j/sqrt(nExamples));
            s(row*28+1:row*28+28,col*28+1:col*28+28) = reshape(data(:,sortIndex(j+1)),28,28);
        end
        filename = strcat('images/resp/l',num2str(l),'u',num2str(i),'.png');
        imwrite(s,filename);
    end
end

