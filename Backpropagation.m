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
        
    end
    
    methods
        
        %   Each training example is a pair of theform (x,t), where x' is the vector of network input
        %   values, and is the vector of target network output values.
        %   q is the learning rate (e.g., .O5). ni, is the number of network inputs, nhidden the number of units in the hidden layer, and no,, the number of output units.
        %   The inputfiom unit i into unit j is denoted xji, and the weightfrom unit i to unit j is denoted wji.
        
        function self=Backpropagation(trainingX,trainingT,eta,Nin,Nout,Nhidden)
            self.trainingX=trainingX;
            self.trainingT=trainingT;
            self.eta=eta;
            self.Nin=Nin;
            self.Nout=Nout;
            self.Nhidden=Nhidden;
            self.errorValue=0.6;
            
            
            %Initialize the weight matrices with random weights
            
            [l,b] = size(trainingX);
            [n,a] = size(trainingT);
            
            self.IHW = rand(l,Nhidden); % Weight matrix from Input to Hidden
            self.HOW = rand(Nhidden,n); % Weight matrix from Hidden to Output
            
            
            
        end
        
        function [errorValue,delta_V,delta_W]=trainNeuralNet(self)
            count=0;
            while self.errorValue > 0.05
                count = count + 1;
                Error_Mat(count)=self.errorValue;
                self.HOW=self.HOW+delta_W;
                self.IHW=self.IHW+delta_V;
                
                count
                
                [errorValue,delta_V,delta_W]=trainNeuralNet(Norm_Input,Norm_Output,self.IHW,self.HOW,delta_V,delta_W);
            end
            
            
            if errorValue < 0.05
                
                %Incrementing count variable to know the number of iteration
                
                count=count+1;
                
                %Storing error value into matrix for plotting the graph
                
                Error_Mat(count)=errorValue;
                
            end
            
            %Calculating error rate
            
            Error_Rate=sum(Error_Mat)/count;
            
            figure;
            
            %setting y value for plotting graph
            
            y=[1:count];
            
            %Plotting graph
            
            plot(y, Error_Mat);
            
            
        end
        
    end
    
    
end