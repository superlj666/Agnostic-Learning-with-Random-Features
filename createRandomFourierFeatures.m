function Z = createRandomFourierFeatures(D, W, b, X,typeRF) 
%CREATERANDOMFOURIERFEATURES creates Gaussian random features
% Inputs:
% D the number of features to make
% W, b the parameters for those features (d x D and 1 x D)
% X the datapoints to use to generate those features (d x N)
    %Z = sqrt(2/D)*cos(bsxfun(@plus,W'*X, b'));
    %Z = sqrt(1/D)*cos(bsxfun(@plus,W'*X, b'));
   % Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
   switch typeRF
       case {'gq','fsir'} 
           Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
       case 'le-rff'
           %{
           n = size(X,2); lambda = 0.05;
           Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
           ZTZ = Z*Z';
           ZTZinv = inv(1/D*ZTZ+n*lambda*eye(2*D));
           M = ZTZ*ZTZinv;
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
             Z = [cos(bsxfun(@plus,w_opm(:,1:D)'*X, b'));sin(bsxfun(@plus,w_opm(:,D+1:end)'*X, b'))];
             Z = repmat(wgh_opm',1,n).*Z;
           %}
           Z = [cos(bsxfun(@plus,W(:,1:D)'*X, b'));sin(bsxfun(@plus,W(:,D+1:end)'*X, b'))];
       otherwise
           Z = sqrt(1/D)*[cos(bsxfun(@plus,W'*X, b'));sin(bsxfun(@plus,W'*X, b'))];
           %Z = sqrt(1/D)*[max(bsxfun(@plus,W'*X, b'),0);max(bsxfun(@plus,W'*X, b'),0)];
   end
end