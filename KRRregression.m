function [accTrain, accTest] = KRRregression(X_train,X_test,Y_train,Y_test,flagCV, kernel)

meany = mean(Y_train);
if flagCV == 1
    L_fold = 5;
    [bestc, ~, ~] = automaticParameterSelectionlambda(Y_train, X_train, L_fold);
    lambda = bestc;
else
    lambda = .5;
end

K = kernel(X_train, X_train);
alpha = (K + lambda * eye (length(Y_train)))\(Y_train-meany);

pred = kernel(X_train, X_train) * alpha;
accTrain = mean(abs(pred - Y_train) .^ 2);

pred = kernel(X_test, X_train) * alpha;
accTest = mean(abs(pred - Y_test) .^ 2);


