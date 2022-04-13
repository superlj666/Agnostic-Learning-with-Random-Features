function [ac] = get_cv_aclambda(y,x,param,nr_fold)
len=length(y);
ac = 0;
rand_ind = randperm(len);
for i=1:nr_fold % Cross training : folding
  test_ind=rand_ind([floor((i-1)*len/nr_fold)+1:floor(i*len/nr_fold)]');
  train_ind = [1:len]';
  train_ind(test_ind) = [];
  meany = mean(y(train_ind));
  w = (x(:,train_ind) * x(:,train_ind)' + param * eye(size(x(:,train_ind),1))) \ (x(:,train_ind) * (y(train_ind)-meany));
  pred = sign(x(:,test_ind)'*w+meany);
  ac = ac + sum(y(test_ind)==pred);
end
ac = ac / len;
fprintf('Cross-validation Accuracy = %g%%\n', ac * 100);
