function [training_time, accTest] = DC_RFclassification(Z_train,Z_test,Y_train,Y_test,flagCV, m)
    M = size(Z_train, 1);
    n = size(Z_train, 2);
    steps = ceil(n/m);
    w = zeros(M, m);
    training_times = zeros(1, m);
    meany = mean(Y_train);
    if flagCV == 1
        L_fold = 5;
        [bestc, ~, ~] = automaticParameterSelectionlambda(Y_train, Z_train, L_fold);
        lambda = bestc;
    else
        lambda = 2^(-4);
    end
    
    for i = 1:m
        if steps * i > n
            part_X = Z_train(:, steps * (i - 1) + 1 : end);
            part_y = Y_train(steps * (i - 1) + 1 : end);
        else
            part_X = Z_train(:, steps * (i - 1) + 1 : steps * i);
            part_y = Y_train(steps * (i - 1) + 1 : steps * i);
        end
        tic;
        w(:,i) = (part_X*part_X' + lambda * eye(M)) \ (part_X * part_y);
        training_times(i) = toc;
    end
    training_time = mean(training_times);

    [err,~, ~] = computeError(Z_test, mean(w, 2), meany, Y_test);
    accTest = (1-err)*100; 
end


