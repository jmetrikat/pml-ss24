include("gaussian.jl")

module Classification
using LinearAlgebra, Distributions
using ..GaussianDistribution
export learn_weight_distribution, prediction
export get_c, get_d, get_e, get_f, get_m, get_new_mu, get_new_sigma, get_rho, get_s_squared, get_tau, get_z, v, w, get_a_new, get_b_new

"""
	learn_logistic_regression_weights(phi, y, epsilon=1e-6)

Learns a bayesian linear classification model for the given feature matrix phi and the labels y and outputs the learned weight distribution (using `epsilon` as a stopping criterion).
Expects phi in in the shape of (i,d), where the number of rows i is the number of samples and the number of columns d is the dimensionality of the feature vector.
Returns a GaussianND.
"""
function learn_weight_distribution(phi::Matrix{Float64}, y::Vector{Float64}; epsilon::Float64 = 1e-6, beta::Float64 = 1.0, tau::Float64 = 1.0)
	# Learn the posterior weight distribution given feature matrix phi and labels y. For the algorithm, see the Unit 8, Slide 11.
	Σ = Matrix(tau^2 * I(size(phi, 2)))
	μ = zeros(size(phi, 2))
	a = fill(0.0, size(phi, 1))
	b = fill(0.0, size(phi, 1))
	a_new = similar(a)
	b_new = similar(b)

	# Repeat until maximum change componentwise in 𝒂 and 𝒃 is small
	while true
		for i in 1:size(phi, 1)
			z = get_z(phi[i, :], Σ)
			e = get_e(phi[i, :], z)
			f = get_f(phi[i, :], μ)
			c = get_c(b[i], e)

			m = get_m(y[i], a[i], f, e, c)
			s_squared = get_s_squared(c, e, beta)

			s = sqrt(s_squared)
			τ = get_tau(s_squared, m, v(m / s, epsilon), w(m / s, epsilon))
			ρ = get_rho(w(m / s, epsilon), s_squared)

			a_new[i] = get_a_new(τ, ρ, beta)
			b_new[i] = get_b_new(ρ, beta)

			d = get_d(b_new[i], b[i], e)

			Σ = get_new_sigma(Σ, b_new[i], b[i], d, z)
			μ = get_new_mu(μ, y[i], a_new[i], a[i], b_new[i], b[i], f, d, z)
		end

		if maximum(abs.(a_new .- a)) < epsilon && maximum(abs.(b_new .- b)) < epsilon
			break
		end

		a = copy(a_new)
		b = copy(b_new)
	end

	return GaussianND(μ, Σ)
end

normal = Distributions.Normal()

# computes the additive correction of a single-sided truncated Gaussian with unit variance
function v(t, epsilon)
	denom = Distributions.cdf(normal, t - epsilon)
	if (denom < floatmin(Float64))
		return (epsilon - t)
	else
		return (Distributions.pdf(normal, t - epsilon) / denom)
	end
end

# computes the multiplicative correction of a single-sided truncated Gaussian with unit variance
function w(t, epsilon)
	denom = Distributions.cdf(normal, t - epsilon)
	if (denom < floatmin(Float64))
		return ((t - epsilon < 0.0) ? 1.0 : 0.0)
	else
		vt = v(t, epsilon)
		return (vt * (vt + t - epsilon))
	end
end

"""
	prediction(phi::Matrix{Float32}, posterior::GaussianND; beta::Float64=1.0)

Calculates predictions (i.e. probability of a sample having the label +1.0) of a logistic regression model with a given posterior weight distribution for a given feature matrix phi.
"""
function prediction(phi::Matrix{Float64}, posterior::GaussianND; beta::Float64 = 1.0)
	# Return predicted probabilities of the labels being +1.0 for a feature matrix phi given the posterior weight distribution.
	return [prediction(phi[i, :], posterior; beta = beta) for i in 1:size(phi, 1)]
end

"""
	prediction(phi::Vector{Float32}, posterior::GaussianND; beta::Float64=1.0)

Calculates prediction (i.e. probability of a sample having the label +1.0) of a logistic regression model with a given posterior weight distribution for a given feature vector phi.
"""
function prediction(phi::Vector{Float64}, posterior::GaussianND; beta::Float64 = 1.0)
	# Return predicted probabily of the label being +1.0 for a feature vector phi given the posterior weight distribution.
	return 1 - cdf(Normal(posterior.mean' * phi, sqrt(beta^2 + phi' * posterior.cov * phi)), 0.0)
end

"""
The following functions are just to help you debugging if you are stuck. Implementing them individually like this is optional.
To use them for debugging, uncomment them as well as their respective test set in test_classification.jl.
"""

function get_z(phi_i, sigma)
	return sigma * phi_i
end

function get_e(phi_i, z)
	return phi_i' * z
end

function get_f(phi_i, mu)
	return phi_i' * mu
end

function get_c(b_i, e)
	return 1 - b_i * e
end

function get_m(y_i, a_i, f, e, c)
	return (y_i * f - a_i * e) / c
end

function get_s_squared(c, e, beta)
	return e / c + beta^2
end

function get_tau(s_squared, m, V, W)
	return (sqrt(s_squared) * V + m * W) / (s_squared * (1 - W))
end

function get_rho(W, s_squared)
	return W / (s_squared * (1 - W))
end

function get_d(b_new, b_i, e)
	return 1 + (b_new - b_i) * e
end

function get_a_new(tau, rho, beta)
	return tau / (1 + rho * beta^2)
end

function get_b_new(rho, beta)
	return rho / (1 + rho * beta^2)
end

function get_new_mu(mu, y_i, a_new, a_i, b_new, b_i, f, d, z)
	return mu + ((y_i * (a_new - a_i) - f * (b_new - b_i)) / d) * z
end

function get_new_sigma(sigma, b_new, b_i, d, z)
	return sigma - ((b_new - b_i) / d) * z * z'
end

end
