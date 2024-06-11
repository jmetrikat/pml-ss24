using Test
include("gaussian.jl")
include("functions.jl")
include("regression.jl")
using .Regression: bayesian_linear_regression
using .GaussianDistribution: GaussianND, Gaussian1DFromMeanVariance, mean, variance
using Plots: plot, plot!, show
using .Functions: compute_features, square, cube

Σ = [1.0 0.0; 0.0 1.0]
Σ_posterior = [4.8045583811559695e-6 -9.221824380022494e-24; -9.221824380022494e-24 6.594976017275093e-8]

@testset "Bayesian Linear Regression Tests" begin
	# generate an array from -10 to 10 with 100 points
	x = range(-10, stop = 10, length = 100)
	x = collect(x)
	y = x .^ 2

	sigma = 1.0

	# create a prior on the weights
	prior_w = GaussianND([0.0, 0.0], Σ)

	# create a set of functions
	functions = [square, cube]

	phi = [compute_features(xi, functions) for xi in x]

	# convert phi to Matrix
	phi = hcat(phi...)'

	# perform Bayesian linear regression
	posterior, predictive = bayesian_linear_regression(phi, y, prior_w, sigma)

	# Check the dimensions of the posterior distribution
	@test size(posterior.mean) == (2,)
	@test size(posterior.cov) == (2, 2)
	# Check if the postererior mean almost equals the expected value
	@test posterior.mean ≈ [0.9999951954416187, 9.221824379954955e-24] atol = 1e-5
	@test posterior.cov ≈ Σ_posterior atol = 1e-15

	# Check the dimensions of the predictive distribution
	@test (mean(predictive(compute_features(5, functions)))) ≈ 24.999879886040468 atol = 1e-5
	@test (variance(predictive(compute_features(5, functions)))) ≈ 1.0040333139909217 atol = 1e-5

end
