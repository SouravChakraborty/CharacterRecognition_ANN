% Load and sort the dataset
table = readtable('original-letter-recognition.dat');
T = sortrows(table,'Letter','ascend');
cell = table2cell(T);

%Finding number of occurences of every letter  in the data. This is being
%done to ensure uniform distribution of training and test data.
x = cell(:,1);
[uniqueX, ~, J] = unique(x);
occ = histc(J, 1:numel(uniqueX));

counter = 1;
trainStartIdx = 1;
temp = 0;

for idx = 1:26
    trainEndIdx = floor(0.7*occ(counter));
    temp = temp + occ(counter);    
    testEndIdx = temp;
        
    if(counter == 1)
        testStartIdx = trainEndIdx + 1;
        TrainData = cell(trainStartIdx:trainEndIdx, :);
        TestData = cell(testStartIdx:testEndIdx, :);
    else
        trainEndIdx = trainStartIdx + trainEndIdx;
        testStartIdx = trainEndIdx + 1;
        tempTrainData = cell(trainStartIdx:trainEndIdx, :);
        tempTestData = cell(testStartIdx:testEndIdx, :);
        
        TrainData = [TrainData;tempTrainData];
        TestData = [TestData;tempTestData];
    end
    
    counter = counter + 1;
    trainStartIdx = testEndIdx + 1;
    end