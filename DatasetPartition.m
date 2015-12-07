function [TrainData, TestData]=DatasetPartition(MagicData, Cell)
[rows,columns]=size(MagicData);
randIdx=randperm(rows);
trainIdx=randIdx(1,1:16000);
testIdx=randIdx(1,16001:20000);

TrainData=Cell(trainIdx,:);

TestData=Cell(testIdx,:);
end
