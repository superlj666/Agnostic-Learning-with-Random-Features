close all
clear all

typeRF = 'sle-rff'; % rff qmc le-rff sle-rff(Ours)
typeC = 'lr'; % ;lr liblinear
flagCV =0;

Iternum =10;
repeate = 10;

M_rates = 0:7;%linspace(1e2, 1e3, 10);
m = 10;


% title = 'EEG';
% sigma = 1;

% title = 'cod-rna';
% sigma = 2;
% 
% title = 'SUSY';%mapstd
% sigma = 32;

% title = 'skin_nonskin';
% sigma = 16;

title = 'covtype';
sigma = 16;

load([title '.mat']);
d = size(X,2);
X = mapstd(X); 
%K = create_kernel(X,X, 'gauss',sigma^2);   

plain_approximate_error = zeros(length(M_rates), repeate);
plain_test_error = zeros(length(M_rates), repeate);
plain_training_time = zeros(length(M_rates), repeate);
plain_generating_time = zeros(length(M_rates), repeate);

leverage_approximate_error = zeros(length(M_rates), repeate);
leverage_test_error = zeros(length(M_rates), repeate);
leverage_training_time = zeros(length(M_rates), repeate);
leverage_generating_time = zeros(length(M_rates), repeate);


for i=1:length(M_rates)
    D = d * 2^M_rates(i);
    
    for j = 1:repeate
        rate = 1/2;
        Training_num = round(length(y)*rate);
        [~, index] = sort(rand( length(y), 1));
        X_train = X( index( end - Training_num+1 : end), : );    
        Y_train = y( index( end - Training_num+1: end));
        X_test = X( index( 1 : end - Training_num), : );
        Y_test = y( index( 1 : end - Training_num));
        
        n=size(X_train,1);

        b = rand(1,D)*2*pi;
        W = RandomFeatures(D, d,sigma,typeRF);
        tic;
        [Z_train_leverage,Z_test_leverage]= SERLSRFF(D, W, b, X_train',X_test',Y_train);       
        leverage_generating_time(i, j)=toc;
        tic;
        Z_train = createRandomFourierFeatures(D, W, b, X_train',typeRF);
        plain_generating_time(i, j)=toc;
        Z_test = createRandomFourierFeatures(D, W, b, X_test',typeRF);
               
        [plain_training_time(i, j), plain_test_error(i, j)] = RFclassification(Z_train,Z_test,Y_train,Y_test,flagCV);
        [leverage_training_time(i, j), leverage_test_error(i, j)] = RFclassification(Z_train_leverage,Z_test_leverage,Y_train,Y_test,flagCV);
     end
    
    fprintf('Plain RF: M=%5d  MSE= %.2f±%.2f, Time= %.6f±%.6f\n', D, mean(plain_test_error(i,:)),std(plain_test_error(i,:)),mean(plain_training_time(i,:)),std(plain_training_time(i,:)));
    fprintf('Leverage RF: M=%5d  MSE= %.2f±%.2f, Time= %.6f±%.6f\n', D, mean(leverage_test_error(i,:)),std(leverage_test_error(i,:)),mean(leverage_training_time(i,:)),std(leverage_training_time(i,:)));
end
save(['./results/rf_leverage', datestr(now,30), '.mat' ]);
% load('./results/rf_leverage20210604T211054.mat');

legend_str = {'Data-dependent RF', 'Uniform RF'};
xlabel_str = '$log_2(M/d)$';

ylabel_str = 'Test accuracy (%)';
Y = {mean(leverage_test_error, 2)', mean(plain_test_error, 2)'};
Z = {std(leverage_test_error, 0, 2)', std(plain_test_error, 0, 2)'};
draw_errorbar_fig(M_rates, Y, Z, ['./results/leverage_accuracy_', title, '.pdf' ], legend_str, xlabel_str, ylabel_str)

ylabel_str = 'Log of training time (s)';
Y = {mean(log(leverage_training_time), 2)', mean(log(plain_training_time), 2)'};
Z = {std(log(leverage_training_time), 0, 2)', std(log(plain_training_time), 0, 2)'};
draw_errorbar_fig(M_rates, Y, Z, ['./results/leverage_training_time_', title, '.pdf' ], legend_str, xlabel_str, ylabel_str)

ylabel_str = 'Log of generating time (s)';
Y = {mean(log(leverage_generating_time), 2)', mean(log(plain_generating_time), 2)'};
Z = {std(log(leverage_generating_time), 0, 2)', std(log(plain_generating_time), 0, 2)''};
draw_errorbar_fig(M_rates, Y, Z, ['./results/leverage_generating_time_', title, '.pdf' ], legend_str, xlabel_str, ylabel_str)