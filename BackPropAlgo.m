function BackPropAlgo(Input, Output)
 
%STEP 1 : Normalize the Input
 
%Checking whether the Inputs needs to be normalized or not
 
if max(abs(Input(:)))> 1 
 
%Need to normalize
 
Norm_Input = Input / max(abs(Input(:)));
 
else
 
Norm_Input = Input;
 
end
 
%Checking Whether the Outputs needs to be normalized or not
 
if max(abs(Output(:))) >1
 
%Need to normalize
 
Norm_Output = Output / max(abs(Output(:)));
 
else
 
Norm_Output = Output;
 
end
 
%Assigning the number of hidden neurons in hidden layer
 
m = 2;
 
%Find the size of Input and Output Vectors
 
[l,b] = size(Input);
 
[n,a] = size(Output);
 
%Initialize the weight matrices with random weights
 
V = rand(l,m); % Weight matrix from Input to Hidden
 
W = rand(m,n); % Weight matrix from Hidden to Output
 
%Setting count to zero, to know the number of iterations
 
count = 0;
 
%Calling function for training the neural network
 
[errorValue delta_V delta_W] = trainNeuralNet(Norm_Input,Norm_Output,V,W);
 
%Checking if error value is greater than 0.1. If yes, we need to train the
 
%network again. User can decide the threshold value
 
while errorValue > 0.05
 
%incrementing count
 
count = count + 1;
 
%Store the error value into a matrix to plot the graph
 
Error_Mat(count)=errorValue;
 
%Change the weight metrix V and W by adding delta values to them
 
W=W+delta_W;
 
V=V+delta_V;
 
%Calling the function with another overload.
 
%Now we have delta values as well.
 
count
 
[errorValue delta_V delta_W]=trainNeuralNet(Norm_Input,Norm_Output,V,W,delta_V,delta_W);
 
end
 
%This code will be executed when the error value is less than 0.1
 
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