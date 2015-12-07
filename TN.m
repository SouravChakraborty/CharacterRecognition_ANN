tblCharRecDS= readtable('original-letter-recognition.csv','ReadVariableNames',false);
C=table2cell(tblCharRecDS);
% Partitioning the dataset 
[TrainData, ValidationData, TestData]=DatasetPartition(tblCharRecDS,C);
% seperating the features and class lables from the dataset
TrainFeatures=TrainData(:,2:17);
TrainFeatures=cell2mat(TrainFeatures);
TrainClass = TrainData(:,1:1);
TrainClass=cell2mat(TrainClass);

TestFeatures = TestData(:,2:17);
TestFeatures=cell2mat(TestFeatures);
TestClassLabels = TestData(:,1:1);
TestClassLabels=cell2mat(TestClassLabels);


%user specified values
hidden_neurons = 3;
epochs = 10000;

% check same number of patterns in each
if size(TrainFeatures,1) ~= size(TrainClass,1)
    disp('ERROR: data mismatch')
   return 
end  

%standardise the data to mean=0 and standard deviation=1
%inputs
mu_inp = mean(TrainFeatures);
sigma_inp = std(TrainFeatures);
TrainFeatures = (TrainFeatures(:,:) - mu_inp(:,1)) / sigma_inp(:,1);

%outputs
TrainClass = TrainClass';
mu_out = mean(TrainClass);
sigma_out = std(TrainClass);
TrainClass = (TrainClass(:,:) - mu_out(:,1)) / sigma_out(:,1);
TrainClass = TrainClass';

%read how many patterns
patterns = size(TrainFeatures,1);
%add a bias as an input
bias = ones(patterns,1);
TrainFeatures = [TrainFeatures bias];

%read how many inputs
inputs = size(TrainFeatures,2);


% ---------- set weights -----------------
%set initial random weights
weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10;
weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10;


%-----------------------------------
%--- Learning Starts Here! ---------
%-----------------------------------

%do a number of epochs
for iter = 1:epochs
    
    %get the learning rate from the slider
    learn_rate=0.01;
    alr=learn_rate*10;
    %loop through the patterns, selecting randomly
    for j = 1:patterns
        
        %select a random pattern
        patnum = round((rand * patterns) + 0.5);
        if patnum > patterns
            patnum = patterns;
        elseif patnum < 1
            patnum = 1;    
        end
       
        %set the current pattern
        this_pat = TrainFeatures(patnum,:);
        act = TrainClass(patnum,1);
        
        %calculate the current error for this pattern
        hval = (tanh(this_pat*weight_input_hidden))';
        pred = hval'*weight_hidden_output';
        error = pred - act;

        % adjust weight hidden - output
        delta_HO = error.*learn_rate .*hval;
        weight_hidden_output = weight_hidden_output - delta_HO';

        % adjust the weights input - hidden
        delta_IH= alr.*error.*weight_hidden_output'.*(1-(hval.^2))*this_pat;
        weight_input_hidden = weight_input_hidden - delta_IH';
        
    end
    % -- another epoch finished
    pred = weight_hidden_output*tanh(TrainFeatures*weight_input_hidden)';
    error = pred' - TrainClass;
    err(iter) =  (sum(error.^2))^0.5;
     %reset weights if requested
%     if reset
%         weight_input_hidden = (randn(inputs,hidden_neurons) - 0.5)/10;
%         weight_hidden_output = (randn(1,hidden_neurons) - 0.5)/10;
%         fprintf('weights reaset after %d epochs\n',iter);
%         reset = 0;
%     end
    
    %stop if error is smalll
    if err(iter) < 0.001
        fprintf('converged at epoch: %d\n',iter);
        break 
    end
       
end