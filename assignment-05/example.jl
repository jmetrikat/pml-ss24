include("bayesian_linear_classification.jl")

using MLDatasets
using ..Classification

"""
	get_mnist_data(class0 = 1, class1 = 9; test=false, nsamples=0)

Loads the first n samples of MNIST test or train data for the classes class0 and class1 and returns it as a feature matrix phi and a label vector y.
"""
function get_mnist_data(class0 = 1, class1 = 9; test = false, nsamples = 0)

	if test == true
		Y = MNIST(split = :test).targets
		X = MNIST(split = :test).features
	else
		Y = MNIST(split = :train).targets
		X = MNIST(split = :train).features
	end

	X0 = reshape(X, (size(X, 1) * size(X, 2), size(X, 3)))[:, Y.==class0]'
	X1 = reshape(X, (size(X, 1) * size(X, 2), size(X, 3)))[:, Y.==class1]'

	if nsamples > 0
		X0 = X0[1:nsamples, :]
		X1 = X1[1:nsamples, :]
	end

	phi = vcat(X0, X1)
	Y = vcat(ones(size(X0, 1)) * -1, ones(size(X1, 1)))

	return Matrix{Float64}(phi), Y
end

data = get_mnist_data(nsamples = 5)

posterior = learn_weight_distribution(data...)

x, y = get_mnist_data(test = true)

prediction_to_label(x) = x > 0.5 ? 1.0 : -1.0

pred_y = prediction_to_label.(prediction(x, posterior))

accuracy = sum(y .== pred_y) / length(y)
println("accuracy: ", accuracy)
