function [mu_shared, delta, W, v, rsltStruct] = NSC_MD_notMissing_final(Y, k, m2, gamma)
%Nearest shrunken centroids for missing data
%This code uses a PPCA formulations along with a prior on class specific
%deviations to learn the parameters for a sparse LDA model. The algorithm
%uses EM such that the input data may contain missing data. For the version
%that handles missing data, please refer to 'NSC_MD.m'
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


%get matrix size
[p,n] = size(Y);

% Initialize convergence criteria
nloglk = inf;
itercount = 0;
MaxIter = 1000; 
TolX = 1e-6;
TolFun = 1e-6;

% Populate initial guess
mu_shared = nanmean(Y,2);
delta = zeros(p,2);
delta(:,1) = nanmean(Y(:,1:m2-1),2) - mu_shared;
delta(:,2) = nanmean(Y(:,m2:end),2) - mu_shared;
W = randn(p,k);
v = rand;

% determine the number of samples per class
n1 = m2 - 1;
n2 = n - m2 + 1;

%initialize storage variables
tau = zeros(p,2);
p_prev = p;

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
    
    % calculate 'M'
    M = v*eye(k) + W'*W;
    
    mu1 = mu_shared + delta(:,1);
    mu2 = mu_shared + delta(:,2);
    
    mu_rep = [repmat(mu1,1,n1), repmat(mu2,1,n2)];
    
    t = M\W'*(Y - mu_rep);
    tt = zeros(k,k,n);
    
    for i = 1:n
        tt(:,:,i) = v.*inv(v*eye(k) + W'*W) + t(:,i)*t(:,i)';
    end
    
    % --------------------- %
    %        M-STEP         %
    % --------------------- %
    
    %update mu
    delta_mat = [delta(:,1)*ones(1,n1), delta(:,2)*ones(1,n2)];
    mu_shared_new = mean((Y - W*t - delta_mat),2);
    
    % update W
    numsum = 0;
    demsum = 0;
    
    for i = 1:n
        numsum = numsum + (Y(:,i) - mu_shared - delta_mat(:,i))*t(:,i)';
        demsum = demsum + tt(:,:,i);
    end
    Wnew = numsum/demsum;
    
    %update delta  
    t_aug1 = 1./(tau(:,1) + v);
    t_aug2 = 1./(tau(:,2) + v);
    delta_new(:,1) = tau(:,1).*t_aug1.*mean(Y(:,1:m2-1) - ...
        W*t(:,1:m2-1) - mu_shared*ones(1,n1),2);
    delta_new(:,2) = tau(:,2).*t_aug2.*mean(Y(:,m2:end) - ...
        W*t(:,m2:end) - mu_shared*ones(1,n2),2);
    
    % update residual variance
    vsum = 0;
    
    for j=1:n
        if j < m2
            munew = mu_shared + delta(:,1);
        else
            munew = mu_shared + delta(:,2);
        end
        
        vsum = vsum + sum(diag(Y(:,j)*Y(:,j)')) - 2*sum(diag(W'*Y(:,j)*t(:,j)')) ...
            + 2*sum(diag(t(:,j)'*W'*munew)) + ...
            sum(diag(tt(:,:,j)*(W'*W))) - 2*Y(:,j)'*munew + munew'*munew;
    end
    vnew = vsum/(n*p);
    
    % --------------------- %
    % Check for convergence %
    % --------------------- %
    
    % Compute negative log-likelihood function
    % find the determinant of the marginal covariance
    Cy_full = Wnew*Wnew' + vnew*eye(p);
    R = [svd(Wnew).^2; zeros(p-k,1)];
    R = sum(log(R + vnew));
    
    mu1 = mu_shared_new + delta_new(:,1);
    mu2 = mu_shared_new + delta_new(:,2);
    mu_rep = [repmat(mu1,1,n1), repmat(mu2,1,n2)];
    Y_aug = Y - mu_rep;
    
    % use vector operations for any observations that do not have missing
    % elemets
    nloglk_data = (n*p*log(2*pi) + n*R + ...
        sum(diag(Cy_full\(Y_aug*Y_aug'))))/2;
    
    % find the negative log likelihood of the prior
    nz_ind1 = tau(:,1) > 10^(-16);
    nz_ind2 = tau(:,2) > 10^(-16);
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
        warning('Negative log likelihood increase')
    end
    
    if norm(delta_new - delta) < TolX*norm(delta)
        break;
    end
    
   %consider the change in observed data log likelihood along with the
   %current degrees of freedom
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
    
end % End of While Loop

% Store additional information at convergence
if nargout > 4
    rsltStruct.NumIter = itercount;
    rsltStruct.nloglk = nloglk;
end

end
