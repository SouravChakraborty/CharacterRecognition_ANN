
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
 
%calculating confusion matrix
[Accuracy,Precision,Recall]=confusionmatrix(ValidationData,dtr);



% Knn method 
Mdl = fitcknn(TrainFeatures,TrainClass,'NumNeighbors',5,'Standardize',1)
predicted_labels = predict(Mdl,TestFeatures);
