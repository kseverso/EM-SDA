function [mu_shared, delta, W, v, rsltStruct] = NSC_MD_final(Y, k, m2, prior)
% Nearest shrunken centroids for missing data
% This code uses a PPCA formulations along with a prior on class specific
% deviations to learn the parameters for a sparse LDA model. The algorithm
% uses EM such that the input data may contain missing data
%
% For full details about the algorithm please refer to "A method for
% learning a sparse classifier in the presence of missing data for
% high-dimensional biological datasets"

% --------------------- %
%        INPUTS         %
% --------------------- %
% 
% Y - p x n matrix of data where n is the number of samples and p is the 
% number of measurements per sample i.e., uses column observations
% k - latent dimension, must be a scalar less than min(n,p) - 1
% m2 - index of the observation where the second class begins
% gamma - value of the parameter than controls the gamma distribution of
% tau; must be a real positive 
%
% --------------------- %
%        OUTPUTS        %
% --------------------- %
%
% mu_shared - p-dimensional vector of the shared mean across both classes
% delta - p-dimensional vector of class-specific deviations from the mean 
% W - p x k matrix of the factor loadings
% v - scalar of the iid noise in the observation
% additional optional outputs:
% itercount - total number of iterations
% nloglk - negative log likelihood at convergence
%
% Copyright (C) 2016, Kristen Severson <kseverso@mit.edu>
%
% This function is free software: you can redistribute it and/or modify it 
% under the terms of the GNU General Public License as published by the 
% Free Software Foundation, either version 3 of the License, or (at your 
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General 
% Public License for more details.
%
% You should have received a copy of the GNU General Public License along 
% with this program. If not, see <http://www.gnu.org/licenses/>.
%

startNaN = isnan(Y);
allNaN = all(startNaN,1);
Y(:,allNaN) = [];
startNaN(:,allNaN)= [];
obs = ~startNaN;

[p,n] = size(Y);

% Initialize convergence criteria
nloglk = inf;
itercount = 0;
MaxIter = 1000;
TolX = 1e-6;
TolFun = 1e-6;

ix = isnan(Y);
Y(ix) = 0;

% Populate initial guess
gamma = prior;
mu_shared = nanmean(Y,2);
delta = zeros(p,2);
delta(:,1) = nanmean(Y(:,1:m2-1),2) - mu_shared;
delta(:,2) = nanmean(Y(:,m2:end),2) - mu_shared;
W = randn(p,k);
v = rand;

% determine the number of samples per class
n1 = m2 - 1;
n2 = n - m2 + 1;
p_prev = p;

%initialize storage variables
x_1 = zeros(k,n);
x_2 = zeros(k,k,n);
y_1 = zeros(p,n);
y_2 = zeros(p,p,n);
xy = zeros(p,k,n);
tau = zeros(p,2);

%populate observed sufficient statistics which will not need to be
%updated
y_1(obs) = Y(obs);

%determine which observations have a full set of measurements
full_ind = find(sum(obs) == p);
n_full = length(full_ind);
full1 = sum(full_ind < m2);
full2 = n_full - full1;

while (itercount < MaxIter)
    itercount = itercount +1;
    
    % --------------------- %
    %        E-STEP         %
    % --------------------- %
    
    % find expectations of the 'taus'
    % need to express expectation in terms of the inverse because we expect
    % that some values of delta will be zero
    tau(:,1) = abs(delta(:,1))./gamma;
    tau(:,2) = abs(delta(:,2))./gamma;
    
    for j = 1:n
        
        if j < m2
            mu = mu_shared + delta(:,1);
        else
            mu = mu_shared + delta(:,2);
        end
        
        %store indices, sizes and observation for current iterate
        idxObs = obs(:,j);
        idxMis = ix(:,j);
        
        %find the number of missing components
        M = sum(idxMis);
        
        %keep the current observation
        y = Y(:,j);
        
        % Use Sherman–Morrison formula to find the inv(v.*eye(k)+w'*w)
        C = eye(k)/v-W(idxObs,:)'*W(idxObs,:)/...
            (eye(k)+W(idxObs,:)'*W(idxObs,:)/v)/(v^2);
        
        % Populate the sufficient statistics for the latent variable
        x_1(:,j) = C*W(idxObs,:)'*(y(idxObs) - mu(idxObs));
        x_2(:,:,j) = v*C + x_1(:,j)*x_1(:,j)';
        
        % Populate the sufficient statistics for the missing varables
        y_1(idxMis,j) = W(idxMis,:)*x_1(:,j) + mu(idxMis);
        
        % During the first iteration, populate the second order sufficient
        % statistic for the observed data
        if itercount == 1
            y_2(idxObs,idxObs,j) = y(idxObs)*y(idxObs)';
        end
        
        % Populate the remaining second order sufficient statistics
        y_2(idxMis,idxMis,j) = v*(eye(M) + W(idxMis,:)*C*W(idxMis,:)') + ...
            y_1(idxMis,j)*y_1(idxMis,j)';
        y_2(idxObs,idxMis,j) = y(idxObs)*y_1(idxMis,j)';
        y_2(idxMis,idxObs,j) = y_1(idxMis,j)*y(idxObs)';
        
        xy(idxObs,:,j) = y(idxObs)*x_1(:,j)';
        xy(idxMis,:,j) = v*W(idxMis,:)*C + y_1(idxMis,j)*x_1(:,j)';
        
    end
    
    % --------------------- %
    %        M-STEP         %
    % --------------------- %
    
    %update mu
    delta_mat = [delta(:,1)*ones(1,n1), delta(:,2)*ones(1,n2)];
    mu_shared_new = mean((y_1 - W*x_1 - delta_mat),2);
    
    %update delta
    t_aug1 = 1./(tau(:,1) + v);
    t_aug2 = 1./(tau(:,2) + v);
    delta_new(:,1) = tau(:,1).*t_aug1.*mean(y_1(:,1:m2-1) - ...
        W*x_1(:,1:m2-1) - mu_shared*ones(1,n1),2);
    delta_new(:,2) = tau(:,2).*t_aug2.*mean(y_1(:,m2:end) - ...
        W*x_1(:,m2:end) - mu_shared*ones(1,n2),2);
    
    % update W
    numsum = 0;
    demsum = 0;
    
    for i = 1:n
        if i < m2
            munew = mu_shared + delta(:,1);
        else
            munew = mu_shared + delta(:,2);
        end
        numsum = numsum + xy(:,:,i) - munew*x_1(:,i)';
        demsum = demsum + x_2(:,:,i);
    end
    Wnew = numsum/demsum;
    
    % update residual variance
    vsum = 0;
    
    for j=1:n
        if j < m2
            munew = mu_shared + delta(:,1);
        else
            munew = mu_shared + delta(:,2);
        end
        
        vsum = vsum + sum(diag(y_2(:,:,j))) - 2*sum(diag(W'*xy(:,:,j))) ...
            + 2*sum(diag(x_1(:,j)'*W'*munew)) + ...
            sum(diag(x_2(:,:,j)*(W'*W))) - 2*y_1(:,j)'*munew + munew'*munew;
    end
    vnew = vsum/(n*p);
    
    % --------------------- %
    % Check for convergence %
    % --------------------- %
    
    % Compute negative log-likelihood function
    % find the determinant of the marginal covariance
    nloglk_data = 0;
    if ~isempty(full_ind)
        Cy_full = Wnew*Wnew' + vnew*eye(p);
        S = [svd(Wnew).^2; zeros(p-k,1)];
        S = sum(log(S + vnew));
        
        mu_rep = [repmat(mu_shared_new + delta_new(:,1),1,full1), ...
            repmat(mu_shared_new + delta_new(:,2),1,full2)];
        Y_aug = Y(:,full_ind) - mu_rep;
        % use vector operations for any observations that do not have missing
        % elemets
        nloglk_data = (p*log(2*pi) + S + ...
            sum(diag(Cy_full\(Y_aug*Y_aug'/n_full))))*n_full/2;
    end
    % find observations that do have missing elements and calculate the
    % negative log likelihood of the observed data
    for m = 1:n
        idxObs = obs(:,m);
        if sum(idxObs) == p
            %do nothing
        else
            if m < m2
                munew = mu_shared_new + delta_new(:,1);
            else
                munew = mu_shared_new + delta_new(:,2);
            end
            
            y = Y(idxObs,m) - munew(idxObs); % the jth observation centered with only complete elements
            Wobs = Wnew(idxObs,:);
            Cy = Wobs*Wobs'+vnew*eye(sum(idxObs));
            Sobs = [svd(Wobs).^2; zeros(sum(idxObs)-k,1)];
            Sobs = sum(log(Sobs + vnew));
            nloglk_data = nloglk_data + ...
                (sum(idxObs)*log(2*pi) + Sobs + trace(Cy\(y*y')))/2;
        end
    end
    
    % find the negative log likelihood of the prior
    nz_ind1 = tau(:,1) > 10^(-16);
    nz_ind2 = tau(:,2) > 10^(-16); %~= 0;
    p1 = sum(nz_ind1);
    p2 = sum(nz_ind2);
    nz_tau1 = tau(nz_ind1, 1);
    nz_tau2 = tau(nz_ind2, 2);
    
    nll_prior = 1/2*sum(log(nz_tau1)) + 1/2*sum(log(nz_tau2)) + ...
        + p1/2*log(2*pi) + p2/2*log(2*pi) - p1*log(gamma/2) - p2*log(gamma/2) + ...
        1/2*sum(delta(nz_ind1,1).^2./nz_tau1) + 1/2*sum(delta(nz_ind2,2).^2./nz_tau2) + ...
        gamma/2*sum(nz_tau1) + gamma/2*sum(nz_tau2);
    
    % add the parts together
    nloglk_new = nll_prior + nloglk_data;
    
    if (nloglk_new > nloglk) && (p1 + p2 == p_prev)
        error(message('Negative log likelihood increase'))
    end
    
    if norm(delta_new - delta) < TolX*norm(delta)
        break;
    end
    
    if (norm(nloglk_new - nloglk) < TolFun*norm(nloglk)) && (p1 + p2 == p_prev)
        break;
    end

    % Update the parameters
    W = Wnew;
    v = vnew;
    mu_shared = mu_shared_new;
    delta = delta_new;
    p_prev = p1 + p2;
    
    %Update the nloglk
    nloglk = nloglk_new;
    
    % Center X
    muX = mean(x_1,2);
    x_1 = bsxfun(@minus,x_1,muX);
    
    % Update the mean of Y, mu
    mu_shared = mu_shared(:,1) + W*muX;
    
end % End of While Loop


% Store additional information at convergence
if nargout > 4
    rsltStruct.NumIter = itercount;
    rsltStruct.nloglk = nloglk;
end

end
