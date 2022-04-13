function [accTrain, accTest] = RFregression(Z_train,Z_test,Y_train,Y_test,flagCV)

meany = mean(Y_train);
if flagCV == 1
    L_fold = 5;
    [bestc, ~, ~] = automaticParameterSelectionlambda(Y_train, Z_train, L_fold);
    lambda = bestc;
else
    lambda = .5;
end
w = (Z_train * Z_train' + lambda * eye(size(Z_train,1))) \ (Z_train * (Y_train-meany));

pred = Z_train'*w;
accTrain = mean(abs(pred - Y_train) .^ 2);

pred = Z_test'*w;
accTest = mean(abs(pred - Y_test) .^ 2);


