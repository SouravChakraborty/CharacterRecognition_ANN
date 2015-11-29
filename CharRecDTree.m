
% Dataset contains both character and numeric data
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


% constructing decision tree 
dtr=fitctree(TrainFeatures,TrainClass,'MinLeafSize',20);
view(dtr,'Mode','Graph');
dtr_predict_lables = predict(dtr,TestFeatures);

%calculating confusion matrix
[TP1,FP1,FN1,TN1]=confusionmatrix(TestClassLabels,dtr_predict_lables);


% Knn method 
Mdl = fitcknn(TrainFeatures,TrainClass,'NumNeighbors',5,'Standardize',1)
knn_predict_lables = predict(Mdl,TestFeatures);
%calculating confusion matrix
[TP2,FP2,FN2,TN2]=confusionmatrix(TestClassLabels,knn_predict_lables);

