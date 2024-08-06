using LinearAlgebra


# Function to compute the radial basis function (RBF) kernel
# X_a: Vector of input values for the first point
# X_b: Vector of input values for the second point
# theta: Tuple of kernel hyperparameters (σ2_f, σ2_n, [λ_1,...,λ_n])
# intuituion
# σ2_f how much the target values vary
# σ2_n noise on each data point
# λ at which distance data points influence each other for each dimension in the data
# Returns: Float64 value representing the kernel evaluation
function rbf_kernel(
	X_a::Vector{<:Real},
	X_b::Vector{<:Real},
	theta::Tuple{Real, Real, Vector},
)::Float64
	σ2_f, σ2_n, λ = theta
	return σ2_f * exp(-0.5 * sum(((X_a[i] - X_b[i])^2 / λ[i]^2) for i in 1:length(X_a))) + σ2_n * (X_a == X_b)
end


# Function to compute the kernel matrix
# X_a: Array of input values for the first set of points
# X_b: Array of input values for the second set of points
# kf: KernelFunction object with a defined kernel function
# theta: Tuple of kernel hyperparameters
# Returns: Kernel matrix C
function kernelmat(X_a::Array, X_b::Array, kernel::Function, theta::Tuple{Real, Real, Vector})
	C = zeros((size(X_a, 1), size(X_b, 1)))
	for i in 1:size(X_a, 1)
		for j in 1:size(X_b, 1)
			C[i, j] = kernel(X_a[i, :], X_b[j, :], theta)
		end
	end
	return C
end


# Define the GP regression model
struct GaussianProcess
	X::Array  # Training inputs
	y::Vector  # Training targets
	kernel::Function #kernel function
	theta::Tuple  # Kernel hyperparameters
	L::Matrix  # Cholesky decomposition of the covariance matrix
end

# HINT you can use LinearAlgebra.Symmetric to ensure that the kernel matrix is symmetric and becomes
# a valid input for LinearAlgebra.cholesky (as it can happen that the kernel matrix is not perfectly symmetric due to numerical errors)
# HINT a data noise (the σ2_n parameter for the rbf kernel) > 0 (as specified by adding it to the diagonal elements of the kernel)
# helps to make the kernel matrix positive semi-definite
# Function to train a Gaussian process
# X: Array of input values
# y: Vector of target values
# kernel: Kernel function
# theta: Tuple of kernel hyperparameters
# Returns: GaussianProcess object
function train_gp(
	X::Array,
	y::Vector,
	kernel::Function,
	theta::Tuple{Real, Real, Vector},
)::GaussianProcess
	C = kernelmat(X, X, kernel, theta)
	# if !isposdef(C)
	# 	C .+= theta[2] * I
	# end
	L = cholesky(Symmetric(C))
	return GaussianProcess(X, y, kernel, theta, L.L)
end

# Function to perform GP prediction
# gp: GaussianProcess object
# X_pred: Array of input values for which to make predictions
# Returns: Tuple of predicted mean and variance
function predict_gp(gp::GaussianProcess, X_pred::Array)
	mean = Vector{Float64}(undef, size(X_pred, 1))
	variance = Vector{Float64}(undef, size(X_pred, 1))

	for i in 1:size(X_pred, 1)
		k = Vector{Float64}(undef, size(gp.X, 1))
		for j in 1:size(gp.X, 1)
			k[j] = gp.kernel(gp.X[j, :], X_pred[i, :], gp.theta)
		end
		mean[i] = k' * (gp.L' \ (gp.L \ gp.y))
		variance[i] = gp.kernel(X_pred[i, :], X_pred[i, :], gp.theta) - (gp.L \ k)' * (gp.L \ k)
	end

	return (mean, variance)
end


# Function to compute the log marginal likelihood
# gp: GaussianProcess object
# Returns: Value representing the log marginal likelihood
function log_m_likelihood(gp::GaussianProcess)
	return -0.5 * (gp.L \ gp.y)' * (gp.L \ gp.y) - sum(log.(diag(gp.L))) - (size(gp.X, 1) / 2) * log(2 * π)
end


# Function to perform gradient descent optimization
# gradient: Function to compute the gradient
# x_0: Initial guess for the optimization
# stepsize: Step size for the optimization; can be a list to set a different step size for each parameter
# (e.g. gradients of σ2_f are a lot smaller than for λ)
# eps: Convergence criterion
# max_iter: Maximum number of iterations
function gradientdescent(
	gradient::Function,
	x_0::Vector,
	Boundaries::Array;
	stepsize::Union{Real, Array} = 1e-3,
	eps::Real = 1e-2,
	max_iter::Int = 200,
)
	status = -1 # OK
	i = 0
	x = x_0

	x_list = []
	push!(x_list, x_0)

	while i <= max_iter
		grad = gradient(x)
		# print(grad, "   ", stepsize .* grad, "   ", x, "\n")
		x_n = x .- stepsize .* grad
		push!(x_list, x_n)

		if any(x_n .< Boundaries[:, 1])
			status = 10
			break
		elseif any(x_n .> Boundaries[:, 2])
			status = -10
			break
		elseif norm(grad) < eps
			status = 1 # OK
			x = x_n
			break
		else
			x = x_n
			i = i + 1
		end
	end
	return (x, x_list, status)
end


# partial derivative of σ2_n set to 0 avoids updating this parameter!
function grad_rbf(x_p, x_q, theta::Tuple{Real, Real, Vector})
	σ2_f, σ2_n, λ = theta

	partial_σ2_f = exp(-(1 / 2) * sum((x_p .- x_q) .^ 2 ./ λ .^ 2))

	partial_σ2_n = 1 * (x_p == x_q)

	partial_λs = Vector{Float64}(undef, size(λ, 1))
	for i ∈ 1:length(λ)
		dist_sq = sum((x_p .- x_q) .^ 2 ./ λ .^ 2)
		partial_λs[i] = σ2_f * exp(-(1 / 2) * dist_sq) * ((x_p[i] - x_q[i])^2 / λ[i]^3)
	end

	return [partial_σ2_f, 0 * partial_σ2_n, partial_λs...]
end


function grad_logmlik(gp::GaussianProcess)

	n_theta = 2 + length(gp.theta[3])

	α = gp.L' \ (gp.L \ gp.y)

	K_inv = inv(gp.L * gp.L')

	K_partial = Array{Float64}(undef, (size(gp.X, 1), size(gp.X, 1), n_theta))

	for i ∈ 1:size(gp.X, 1)
		for j ∈ 1:size(gp.X, 1)
			K_partial[i, j, :] = grad_rbf(gp.X[i, :], gp.X[j, :], gp.theta)
		end
	end

	grad = Vector{Float64}(undef, n_theta)
	for i ∈ 1:size(K_partial, 3)
		grad[i] = -0.5 * tr((α * α' - K_inv) * K_partial[:, :, i])
	end

	return grad
end


function optimize_theta(
	X::Array,
	y::Vector,
	theta_init::Tuple;
	stepsize::Union{Real, Array} = 1e-3,
	eps::Real = 1e-2,
	max_iter::Int = 200,
)

	n_theta = 2 + length(theta_init[3])

	function gradient(theta)

		theta = (theta[1], theta[2], theta[3:end])

		gp = train_gp(X, y, rbf_kernel, theta)

		return grad_logmlik(gp)
	end


	bounds = zeros((n_theta, 2))
	bounds[:, 2] .= 10^20

	theta_star, thetas_list, status = gradientdescent(gradient, vcat(theta_init...), bounds; stepsize, eps, max_iter)

	println("Status: ", status)

	return ((theta_star[1], theta_star[2], theta_star[3:end]), thetas_list)
end
