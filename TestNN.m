load house_dataset












% load simpleclass_dataset;
% net=patternnet(7);
% net.trainFcn='traingd';
% net.trainParam.epochs=1000;
% net.trainParam.lr=0.05;
% net.performFcn='mse';
% net.divideFcn='dividerand';
% net.divideParam.trainRatio=70/100;
% net.divideParam.valRatio=15/100;
% net.divideParam.testRatio=15/100;
% 
% %#training 
% net=init(net);
% [net,tr]=train(net,simpleclassInputs,simpleclassTargets); 
% y_hat = net(simpleclassInputs);
% perf = perform(net, simpleclassTargets, y_hat);
% err = gsubtract(simpleclassTargets, y_hat);
% view(net)
