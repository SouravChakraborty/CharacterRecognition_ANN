% Dataset contains both character and numeric data
tblCharRecDS= readtable('original-letter-recognition.csv','ReadVariableNames',false);
C=table2cell(tblCharRecDS);

%Partitioning the dataset 
[TrainData,TestData]=DatasetPartition(tblCharRecDS,C);

% seperating the features and classlables from the dataset
%%% Train data features and classlabels seperation %%% 
%%% trainX - training features & trainT- training classlabesl
trainX=TrainData(:,2:17);
trainT = TrainData(:,1:1);
trainX=cell2mat(trainX);
trainT=cell2mat(trainT);


%%% Test data features and classlabels seperation %%%%
testX = TestData(:,2:17);
testX=cell2mat(testX);
testT = TestData(:,1:1);
testT=cell2mat(testT);

%%% Conversion of categorical classlabels to numericlabels %%%%
trainT=double(trainT)-64;
testT=double(testT)-64;



% BackPropAlgo(TestFeatures,TestClassLabels);

% Pattern Recognition with neural network 
newTrainX=trainX';

newTrainT=full(ind2vec(double(trainT')));


% % % % % % % % % % % %  Backpropagation setting parameters 
 eta = 0.05;
 Nin = 16;
 Nhidden = 12;
 Nout = 1;


% obj=Backpropagation(newTrainX,trainT,eta,Nin,Nout,Nhidden,testX,testT)
% % 
% obj.BackPropagationANN();
% 

% Pattern Recognition with neural network 

%random seed is set to avoid this randomness.
setdemorandstream(391418381);
net = patternnet(30); % taking 30 hidden nodes

% net.trainParam.max_fail=30;
%training the neural network 
[net,tr] = train(net,newTrainX,newTrainT);

nntraintool
plotperform(tr);


%testing the neural network with test data
newTestX = newTrainX(:,tr.testInd);
newTestT = newTrainT(:,tr.testInd);

newTestY = net(newTestX);
testIndices = vec2ind(newTestY);

newTestYT=full(ind2vec(double(testIndices)));
 plotconfusion(newTestT,newTestY)
[c,cm] = confusion(newTestT,newTestYT);
fprintf('performance measures by backpropogation tree classification\n');
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);



trOut = vec2ind(net(newTrainX(:,tr.trainInd)));
vOut = vec2ind(net(newTrainX(:,tr.valInd)));
tsOut = vec2ind(net(newTrainX(:,tr.testInd)));

trTarg = vec2ind(newTrainT(:,tr.trainInd));
vTarg = vec2ind(newTrainT(:,tr.valInd));
tsTarg = vec2ind(newTrainT(:,tr.testInd));
figure();
plotregression(trTarg,trOut,'Train',vTarg,vOut,'Validation',...
tsTarg,tsOut,'Testing')




% Verifying classification performance measures with Decision Tree
dtr=fitctree(trainX,trainT,'MinLeafSize',50);
% view(dtr,'Mode','Graph');
testY = predict(dtr,testX);

dtrTestX=testX';

dtrTestT=full(ind2vec(double(testT')));
dtrTestY=full(ind2vec(double(testY')));



% plotconfusion(dtrTestT,dtrTestY)
[c,cm] = confusion(dtrTestT,dtrTestY);
fprintf('performance measures by Decision tree classification\n');
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);



% Verifying classification performance measures with KNN
Mdl = fitcknn(trainX,trainT,'NumNeighbors',120,'Standardize',1);
testY = predict(Mdl,testX);

knnTestX=testX';
knnTestT=full(ind2vec(double(testT')));
knnTestY=full(ind2vec(double(testY')));

% plotconfusion(dtrTestT,dtrTestY)
[c,cm] = confusion(knnTestT,knnTestY);
fprintf('performance measures by KNN classification\n');
fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);


