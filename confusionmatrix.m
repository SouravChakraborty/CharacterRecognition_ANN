function [Accuracy,Precision,Recall]=confusionmatrix(TrainData,dtr)

TestFeatures = TrainData(:,2:17);
TestFeatures=cell2mat(TestFeatures);
TargetClassLabels = TrainData(:,1:1);
TargetClassLabels=cell2mat(TargetClassLabels);

Prediction = predict(dtr,TestFeatures);

TP=0;FP=0;FN=0;TN=0;
for idx=1:length(TargetClassLabels)
    if(TargetClassLabels(idx,1)==Prediction(idx,1))
        TP=TP+1;
    else
        TN=TN+1;                  %FP(length(FP)+1,1)=Target(:,1);
    end
    
end
% calculating probabilities for validation data

P=TP+FN;
N=FP+TN;

Accuracy=(TP+TN)/(P+N);
Precision=TP/(TP+FP);
Recall=TP/(TP+FN);

ConfMat=[TP,FP,FN,TN,Accuracy,Precision,Recall];

end