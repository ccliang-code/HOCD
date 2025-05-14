function [K,elapse] = constructKernel(fea_a,fea_b,options)
% function [K,elapse] = constructKernel(fea_a,fea_b,options)
%	Usage:
%	K = constructKernel(fea_a,[],options)
%
%   K = constructKernel(fea_a,fea_b,options)
%
%	fea_a, fea_b  : Rows of vectors of data points. 
%
%   options       : Struct value in Matlab. The fields in options that can
%                   be set: 
%           KernelType  -  Choices are:
%               'Gaussian'      - e^{-(|x-y|^2)/2t^2}
%               'Polynomial'    - (x'*y)^d
%               'PolyPlus'      - (x'*y+1)^d
%
%               t       -  parameter for Gaussian
%               d       -  parameter for Poly
%
%   Example:    Ktest = constructKernel(feaTest,fea,options)
%
%   Written by Xiaoli Jiao (jiao AT cs.umbc.edu)
%

if (~exist('options','var'))
   options = [];
else
   if ~strcmpi(class(options),'struct') 
       error('parameter error!');
   end
end



%=================================================
if ~isfield(options,'KernelType')
    options.KernelType = 'Gaussian';
end

switch lower(options.KernelType)
    case {lower('Gaussian')}        %  e^{-(|x-y|^2)/2t^2}
        if ~isfield(options,'t')
            options.t = 1
            %c=162
        end
    case {lower('Polynomial')}      % (x'*y)^d
        if ~isfield(options,'d')
            %options.d = 2;
            options.d = 3
        end
    case {lower('PolyPlus')}      % (x'*y+1)^d
        if ~isfield(options,'d')
            options.d = 3
        end
    otherwise
        error('KernelType does not exist!');
end


tmp_T = cputime;
%=================================================

switch lower(options.KernelType)
    case {lower('Gaussian')}       
        if isempty(fea_b)
            D = EuDist2(fea_a,[],0);
        else
            D = EuDist2(fea_a,fea_b,0);
        end     
        c=2*options.t^2;
        K = exp(-D/c);
        
        %K = exp(-D);
    case {lower('Polynomial')}     
        if isempty(fea_b)
            D = fea_a * fea_a';
        else
            D = fea_a * fea_b';
        end
        K = D.^options.d;
    case {lower('PolyPlus')}     
        if isempty(fea_b)
            D = fea_a * fea_a';
        else
            D = fea_a * fea_b';
        end
        K = (D+1).^options.d;
    otherwise
        error('KernelType does not exist!');
end

if isempty(fea_b)
    K = max(K,K');
end

elapse = cputime - tmp_T;
end
%%
function D = EuDist2(fea_a,fea_b,bSqrt)
%EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
%Matlab matrix operations.
%
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%
%    Examples:
%
%       a = rand(500,10);
%       b = rand(1000,10);
%
%       A = EuDist2(a); % A: 500*500
%       D = EuDist2(a,b); % D: 500*1000
%
%   version 2.1 --November/2011
%   version 2.0 --May/2009
%   version 1.0 --November/2005
%
%   Written by Deng Cai (dengcai AT gmail.com)


if ~exist('bSqrt','var')
    bSqrt = 1;
end

if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';
    
    if issparse(aa)
        aa = full(aa);
    end
    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';

    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end

    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end
end