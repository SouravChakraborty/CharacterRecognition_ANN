function [TrainData, ValidationData, TestData]=DatasetPartition(MagicData, Cell)
[rows,columns]=size(MagicData);
randIdx=randperm(rows);
trainIdx=randIdx(1,1:12000);
validationIdx=randIdx(1,12001:16000);
testIdx=randIdx(1,16001:20000);

TrainData=Cell(trainIdx,:);
ValidationData=Cell(validationIdx,:);
TestData=Cell(testIdx,:);
end
