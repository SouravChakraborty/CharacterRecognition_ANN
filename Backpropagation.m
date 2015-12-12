classdef Backpropagation
    properties
        trainingX=[];
        trainingT=[];
        eta=0.05;
        Nin=0;
        Nout=0;
        Nhidden=0;
        IHW=[];
        HOW=[];
        errorValue=0;
        nargin = 4;
        
    end
    
    methods
        
        %   Each training example is a pair of theform (x,t), where x' is the vector of network input
        %   values, and is the vector of target network output values.
        %   q is the learning rate (e.g., .O5). ni, is the number of network inputs, nhidden the number of units in the hidden layer, and no,, the number of output units.
        %   The inputfiom unit i into unit j is denoted xji, and the weightfrom unit i to unit j is denoted wji.
        
        function self=Backpropagation(trainingX,trainingT,eta,Nin,Nout,Nhidden)
            
            for indx=1:size(trainingX,1)
                trainingX(indx,:) = trainingX(indx,:)/ max(abs(trainingX(indx,:)));
            end
            if max(abs(trainingT(:))) >1
                
                trainingT = trainingT / max(abs(trainingT(:)));
                
            end
            
            
            self.trainingX=trainingX;
            self.trainingT=trainingT';
            self.eta=eta;
            self.Nin=Nin;
            self.Nout=Nout;
            self.Nhidden=Nhidden;
            self.errorValue=0.6;
            self.nargin = 4;
            
            %Initialize the weight matrices with random weights
            
            [l,b] = size(self.trainingX);
            [n,a] = size(self.trainingT);
            
            self.IHW = rand(l,Nhidden); % Weight matrix from Input to Hidden  V 16 X 30
            self.HOW = rand(Nhidden,n); % Weight matrix from Hidden to Output W 30 X 26
            
            
            
        end
        
        function  BackPropagationANN(self)
            
            %---Setting the training parameters
            iterations = 5000;
            errorThreshhold = 0.1;
            learningRate = 0.5;
            %---Set hidden layer type, for example: [4, 3, 2]
            hiddenNeurons = [12];
            
            %---'Xor' training data
%             trainInp = [0 0; 0 1; 1 0; 1 1];
%             trainOut = [0; 1; 1; 0];

            trainInp=self.trainingX';
             trainOut = self.trainingT';
            testInp = trainInp;
            testRealOut =self.trainingT';
            
            %---Initialize Network attributes
            inArgc = size(trainInp, 2);
            outArgc = size(trainOut, 2);
            trainsetCount = size(trainInp, 1);
            
            %---Add output layer
            layerOfNeurons = [hiddenNeurons, outArgc];
            layerCount = size(layerOfNeurons, 2);
            
            %---Weight and bias random range
            e = 1;
            b = -e;
            
            %---Set initial random weights
            weightCell = cell(1, layerCount);
            for i = 1:layerCount
                if i == 1
                    weightCell{1} = unifrnd(b, e, inArgc,layerOfNeurons(1));
                else
                    weightCell{i} = unifrnd(b, e, layerOfNeurons(i-1),layerOfNeurons(i));
                end
            end
            %---Set initial biases
            biasCell = cell(1, layerCount);
            for i = 1:layerCount
                biasCell{i} = unifrnd(b, e, 1, layerOfNeurons(i));
            end
            
            %----------------------
            %---Begin training
            %----------------------
            for iter = 1:iterations
                for i = 1:trainsetCount
                    choice = i;
                    sampleIn = trainInp(choice, :);
                    sampleTarget = trainOut(choice, :);
                    [realOutput, layerOutputCells] = self.ForwardNetwork(sampleIn, layerOfNeurons, weightCell, biasCell);
                    [weightCell, biasCell] = self.BackPropagate(learningRate, sampleIn, realOutput, sampleTarget, layerOfNeurons, ...
                        weightCell, biasCell, layerOutputCells);
                end
                %plot overall network error at end of each iteration
                error = zeros(trainsetCount, outArgc);
                for t = 1:trainsetCount
                    [predict, layeroutput] = self.ForwardNetwork(trainInp(t, :), layerOfNeurons, weightCell, biasCell);
                    p(t) = predict;
                    error(t, : ) = predict - trainOut(t, :);
                end
                err(iter) = (sum(error.^2)/trainsetCount)^0.5;
              err
                %---Stop if reach error threshold
                if err(iter) < errorThreshhold
                    break;
                end
            end
            
            %--Test the trained network with a test set
            testsetCount = size(testInp, 1);
            error = zeros(testsetCount, outArgc);
            for t = 1:testsetCount
                [predict, layeroutput] = self.ForwardNetwork(testInp(t, :), layerOfNeurons, weightCell, biasCell);
                p(t) = predict;
                error(t, : ) = predict - testRealOut(t, :);
            end
            %---Print predictions
            fprintf('Ended with %d iterations.\n', iter);
            a = testInp;
            b = testRealOut;
            c = p';
            x1_x2_act_pred_err = [a b c c-b]
            %---Plot Surface of network predictions
            testInpx1 = [-1:0.1:1];
            testInpx2 = [-1:0.1:1];
            [X1, X2] = meshgrid(testInpx1, testInpx2);
            testOutRows = size(X1, 1);
            testOutCols = size(X1, 2);
            testOut = zeros(testOutRows, testOutCols);
            for row = [1:testOutRows]
                for col = [1:testOutCols]
                    test = [X1(row, col), X2(row, col)];
                    [out, l] = self.ForwardNetwork(test, layerOfNeurons, weightCell, biasCell);
                    testOut(row, col) = out;
                end
            end
            figure(2);
            surf(X1, X2, testOut);
        end
        
        %% BackPropagate: Backpropagate the output through the network and adjust weights and biases
        function [weightCell, biasCell] = BackPropagate(self,rate, in, realOutput, sampleTarget, layer, weightCell, biasCell, layerOutputCells)
            layerCount = size(layer, 2);
            delta = cell(1, layerCount);
            D_weight = cell(1, layerCount);
            D_bias = cell(1, layerCount);
            %---From Output layer, it has different formula
            output = layerOutputCells{layerCount};
            delta{layerCount} = output .* (1-output) .* (sampleTarget - output);
            preoutput = layerOutputCells{layerCount-1};
            D_weight{layerCount} = rate .* preoutput' * delta{layerCount};
            D_bias{layerCount} = rate .* delta{layerCount};
            %---Back propagate for Hidden layers
            for layerIndex = layerCount-1:-1:1
                output = layerOutputCells{layerIndex};
                if layerIndex == 1
                    preoutput = in;
                else
                    preoutput = layerOutputCells{layerIndex-1};
                end
                weight = weightCell{layerIndex+1};
                sumup = (weight * delta{layerIndex+1}')';
                delta{layerIndex} = output .* (1 - output) .* sumup;
                D_weight{layerIndex} = rate .* preoutput' * delta{layerIndex};
                D_bias{layerIndex} = rate .* delta{layerIndex};
            end
            %---Update weightCell and biasCell
            for layerIndex = 1:layerCount
                weightCell{layerIndex} = weightCell{layerIndex} + D_weight{layerIndex};
                biasCell{layerIndex} = biasCell{layerIndex} + D_bias{layerIndex};
            end
        end
        
        
        %% ForwardNetwork: Compute feed forward neural network, Return the output and output of each neuron in each layer
        function [realOutput, layerOutputCells] = ForwardNetwork(self,in, layer, weightCell, biasCell)
            layerCount = size(layer, 2);
            layerOutputCells = cell(1, layerCount);
            out = in;
            for layerIndex = 1:layerCount
                X = out;
                bias = biasCell{layerIndex};
                out = self.Sigmoid(X * weightCell{layerIndex} + bias);
                layerOutputCells{layerIndex} = out;
            end
            realOutput = out;
        end
        
        function y = Sigmoid(self,x)
            y = 1./ (1 + exp(-x));
        end
        
        
        
    end
    
    
end