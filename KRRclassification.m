function [lambda, accTest] = KRRclassification(X_train,X_test,Y_train,Y_test,flagCV, kernel)

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

pred = sign(kernel(X_test, X_train) * alpha+meany);
error = mean(pred~=Y_test);
accTest = (1-error)*100;


