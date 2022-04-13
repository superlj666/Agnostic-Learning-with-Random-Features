function [Z_train,Z_test] = leverageRFF(D, W, b, X,XT)

n = size(X,2); lambda = 0.05;
Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
%Z = [cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
ZTZ = Z*Z';

clear Z

ZTZinv = inv(ZTZ+n*lambda*eye(2*D));
M = ZTZ*ZTZinv;

clear ZTZ ZTZinv

l = trace(M);
n_feat_draw = 2*D;
pi_s = diag(M);
qi_s = pi_s/l;
is_wgt = sqrt(1./qi_s);
[~,wgt_order]=sort(is_wgt,'ascend');
w_order = wgt_order((2*D-n_feat_draw+1):end);
w_opm = zeros(size(X,1),n_feat_draw);
wgh_opm = zeros(1,n_feat_draw);
WW = [W,W];
for ii = 1:n_feat_draw
    order = w_order(ii);
    w_opm(:,ii) = WW(:,order);
    wgh_opm(ii) = is_wgt(order);
end
%wgh_opm = wgh_opm/sum(wgh_opm);
%W_train = repmat(wgh_opm',1,size(X,1)).*w_opm';
Z = sqrt(1/D)*[cos(bsxfun(@plus,w_opm(:,1:D)'*X, b'));sin(bsxfun(@plus,w_opm(:,D+1:end)'*X, b'))];
Z_train = repmat(wgh_opm',1,n).*Z;

clear Z

Z1 = sqrt(1/D)*[cos(bsxfun(@plus,w_opm(:,1:D)'*XT, b'));sin(bsxfun(@plus,w_opm(:,D+1:end)'*XT, b'))];
Z_test = repmat(wgh_opm',1,size(XT,2)).*Z1;


