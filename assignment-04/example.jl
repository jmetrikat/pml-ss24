include("gaussian.jl")
include("functions.jl")
include("regression.jl")
using .Regression: bayesian_linear_regression
using .GaussianDistribution: GaussianND, Gaussian1DFromMeanVariance, mean, variance
using Plots: plot, plot!, show, savefig
using .Functions: compute_features, square, cube
using LinearAlgebra

function run_bayesian_regression(actual)
	# generate an array from -10 to 10 with 100 points
	x = range(-10, stop = 10, length = 100)
	x = collect(x)
	y = actual(x)

	sigma = 1.0

	# create a prior on the weights
	prior_w = GaussianND([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], Matrix{Float64}(I, 6, 6))

	# create a set of functions
	functions = [square, cube, exp, sin, cos, abs]

	phi = [compute_features(xi, functions) for xi in x]

	# convert phi to Matrix
	phi = hcat(phi...)'

	# perform Bayesian linear regression
	posterior, predictive = bayesian_linear_regression(phi, y, prior_w, sigma)

	# plot predictive distribution
	y_pred = [predictive(phi[i, :]) for i in 1:size(phi, 1)]

	# Plot the data and the predictive distribution
	plt = plot(x, y, label = "Data")  # plot the data

	# Extrahieren Sie Mittelwerte und Varianzen aus y_pred
	means = [mean(dist) for dist in y_pred]
	variances = [variance(dist) for dist in y_pred]

	# Fügen Sie Mittelwerte und Varianzband zum Plot hinzu
	plot!(plt, x, means, ribbon = variances, fillalpha = 0.3, label = "Predictive distribution")
	savefig(plt, "bayesian_regression.png")
end

# Example usage:
function actual(x)
	return x .^ 2
end

run_bayesian_regression(actual)
