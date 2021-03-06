classdef Backpropagation
    properties
%         Parameters of backpropagation 
        trainingX=[];
        trainingT=[];
        testX=[];
        testT=[];
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
        %   Initialize the object with traning attributes, training classes
        %   no. of input,hidden and output neurons and test attributes and
        %   test target classes
       
        function self=Backpropagation(trainingX,trainingT,eta,Nin,Nout,Nhidden,testX,testT)
            
            for indx=1:size(trainingX,1)
                trainingX(indx,:) = trainingX(indx,:)/ max(abs(trainingX(indx,:)));
            end
            if max(abs(trainingT(:))) >1
                trainingT = trainingT / max(abs(trainingT(:)));
            end
            
            for indx=1:size(testX,1)
                testX(indx,:) = testX(indx,:)/ max(abs(testX(indx,:)));
            end
            if max(abs(testT(:))) >1
                testT = testT / max(abs(testT(:)));
            end
            
       
            self.trainingX=trainingX;
            self.trainingT=trainingT';
            
            self.testX=testX';
            self.testT=testT';
            self.eta=eta;
            self.Nin=Nin;
            self.Nout=Nout;
            self.Nhidden=Nhidden;
            self.errorValue=0.6;
            self.nargin = 4;
            
            [l,b] = size(self.trainingX);
            [n,a] = size(self.trainingT);
            
            self.IHW = rand(l,Nhidden); % Weight matrix from Input to Hidden  V 16 X 30
            self.HOW = rand(Nhidden,n); % Weight matrix from Hidden to Output W 30 X 26
       
        end
        
        function  BackPropagationANN(self)
            
            %---Set training constraints
            iterations = 10;
            errorThreshhold = 0.1;
            learningRate = 0.5;
            %---Set hidden layer type: 1 hidden layer with 12 neurons 
            hiddenNeurons = [12];
            
            
            trainInp=self.trainingX';
            trainOut = self.trainingT';
            testInp = self.testX';
            testRealOut =self.testT';
            
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
            

            %---Start training 
            
            for iter = 1:iterations
                for i = 1:trainsetCount
                    choice = i;
                    sampleIn = trainInp(choice, :);
                    sampleTarget = trainOut(choice, :);
                    [realOutput, layerOutputCells] = self.ForwardNetwork(sampleIn, layerOfNeurons, weightCell, biasCell);
                    [weightCell, biasCell] = self.BackPropagate(learningRate, sampleIn, realOutput, sampleTarget, layerOfNeurons, ...
                        weightCell, biasCell, layerOutputCells);
                end
                
                error = zeros(trainsetCount, outArgc);
                for t = 1:trainsetCount
                    [predict, layeroutput] = self.ForwardNetwork(trainInp(t, :), layerOfNeurons, weightCell, biasCell);
                    p(t) = predict;
                    error(t, : ) = predict - trainOut(t, :);
                end
                err(iter) = (sum(error.^2)/trainsetCount)^0.5;
              err
              
                %---break the loop when the error is less than threshold
                if err(iter) < errorThreshhold
                    break;
                end
            end
            figure();
             plot(err);
             
            %--Test the trained network with a test set
            testsetCount = size(testInp, 1);
            error = zeros(testsetCount, outArgc);
            for t = 1:testsetCount
                [predict, layeroutput] = self.ForwardNetwork(testInp(t, :), layerOfNeurons, weightCell, biasCell);
                q(t) = predict;
                error(t, : ) = predict - testRealOut(t, :);
            end
            
           
            %---Calculate the accuracy for backpropagation 
            fprintf('Ended with %d iterations.\n', iter);
            a = testInp;
            b = testRealOut;
            c = q';
            b=b*26+64;
            c=c*26+64;
            
            AC=char(b);
            PC=char(ceil(c));
            AC=double(AC)-64;
            PC=double(PC)-64;
            
            newAC=full(ind2vec(double(AC')));
            newPC=full(ind2vec(double(PC')));
            indAC=vec2ind(newAC);
            indPC=vec2ind(newPC);
            newPCYT=full(ind2vec(double(indPC)));
            newACYT=full(ind2vec(double(indAC)));
            [c,cm] = confusion(newACYT,newPCYT)
            fprintf('performance measures by backpropogation tree classification\n');
fprintf('Percentage Correct Classification   : %f%%\n', 100*(c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*(1-c));

        end
        
     
        % backpropagaton of the output through the network with adjusted
        % weights and biases
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
        
        %compute feed farwardnetwork
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