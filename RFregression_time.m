function [accTest, time] = RFregression_time(Z_train,Z_test,Y_train,Y_test,flagCV)

meany = mean(Y_train);
if flagCV == 1
    L_fold = 5;
    [bestc, ~, ~] = automaticParameterSelectionlambda(Y_train, Z_train, L_fold);
    lambda = bestc;
else
    lambda = .05;
end
tic;
w = (Z_train * Z_train' + lambda * eye(size(Z_train,1))) \ (Z_train * (Y_train-meany));
time = toc;

pred = Z_test'*w;
accTest = mean(abs(pred - Y_test) .^ 2);


