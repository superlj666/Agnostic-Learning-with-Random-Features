function W = RandomFeatures(D, d,sigma,typeRF)

switch typeRF
    case {'rff','le-rff','sle-rff'}
        W = randn(d,D)/sigma;
        
    case 'qmc'
        %   scram: scramble the sequence or not
        %   skip, leap: properties in sequences that might affect the final accuracy
        leap = 700; skip = 1000; scram = 1;
        sequence = 'halton';
        switch sequence
            case 'halton'
                p = haltonset(d,'Skip',skip,'Leap',leap);
                if scram p = scramble(p,'RR2'); end
                points = p(1:D,1:d);
            case 'sobol'
                p = sobolset(d,'Skip',skip,'Leap',leap);
                if scram p = scramble(p,'MatousekAffineOwen'); end
                points = p(1:D,1:d);
            case 'lattice'
                latticeseq_b2('initskip'); % see http://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/
                points = latticeseq_b2(d,D)';
                if scram points = mod1shift(points', rand(d,1)); points = points'; end
            case 'digit'
                load new-joe-kuo-5.21201.Cmat % see http://people.cs.kuleuven.be/~dirk.nuyens/qmc-generators/
                digitalseq_b2g('initskip', 18, new_joe_kuo_5_21201);
                points = digitalseq_b2g(d,D)';
                if scram points = digitalshift(points', rand(d,1)); points = points'; end
        end
        
        W = norminv(points', 0, 1);
        
end

