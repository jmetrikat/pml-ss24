
include("gaussian.jl")


module Regression
using ..GaussianDistribution: GaussianND, Gaussian1D, Gaussian1DFromMeanVariance
export bayesian_linear_regression

"""
	bayesian_linear_regression(x, y, prior_w, functions)

Performs Bayesian linear regression on the data `x` (already transformed into feature space and named `phi`) and `y` with a prior on the weights `prior_w` and noise parameter `sigma``.
"""
function bayesian_linear_regression(phi, y, prior_w::GaussianND, sigma)
	μ = prior_w.mean
	Σ = prior_w.cov
	S = inv(inv(Σ) + (1 / sigma^2) * phi'phi)

	posterior_mean = S * (inv(Σ) * μ + (1 / sigma^2) * phi'y)
	posterior_cov = S

	function predictive(phi)
		mean = posterior_mean' * phi
		variance = sigma^2 + phi' * posterior_cov * phi
		return Gaussian1DFromMeanVariance(mean, variance)
	end

	return GaussianND(posterior_mean, posterior_cov), predictive
end

end
