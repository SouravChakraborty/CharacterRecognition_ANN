function [TP,FP,FN,TN]=confusionmatrix(testclasses,predictclasses)

TP=0;FP=0;FN=0;TN=0;
for idx=1:length(testclasses)
    if(testclasses(idx,1)==predictclasses(idx,1))
        TP=TP+1;
    else
        TN=TN+1;                  %FP(length(FP)+1,1)=Target(:,1);
    end
    
end

end