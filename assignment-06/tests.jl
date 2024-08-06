include("GP.jl")
using Distributions

using Test

@testset "RBF Kernel implementation" begin
	x1 = [1, 2, 6, 8, 10]
	x2 = [1, 2, 6, 8, 5]
	theta = (2, 2, [2, 2, 2, 2, 2])
	result = rbf_kernel(x1, x2, theta)
	@test isapprox(result, 0.08787386724681484)
end

@testset "RBF Kernel implementation (different values)" begin
	x1 = [3, -5, 4, 2, 20]
	x2 = [3.1, -5.1, 3.9, 2.1, 19]
	theta = (3, 9, [0.3, 1.5, 2, 0.1, 0.5])
	result = rbf_kernel(x1, x2, theta)
	@test isapprox(result, 0.232139803129576)
end

@testset "kernelmat implementation" begin
	x1 = [1, 2, 6, 8, 10]
	x2 = [1, 2, 6, 8, 5]
	theta = (2, 0.000001, [2])
	C = kernelmat(x1, x2, rbf_kernel, theta)
	@test size(C) == (5, 5)
	@test isapprox(2.000001, C[1, 1])
	@test isapprox(1.764993805169191, C[1, 2])
	@test isapprox(1.764993805169191, C[2, 1])
	@test isapprox(0.08787386724681484, C[1, 3])
	@test isapprox(0.08787386724681484, C[3, 1])
end

@testset "kernelmat implementation (different theta)" begin
	x1 = [1, 2, 6, 8, 10]
	x2 = [1, 2, 6, 8, 5]
	theta = (1.5, 0.000001, [2])
	C = kernelmat(x1, x2, rbf_kernel, theta)
	@test size(C) == (5, 5)
	@test isapprox(1.500001, C[1, 1])
	@test isapprox(1.3237453538768933, C[1, 2])
	@test isapprox(1.3237453538768933, C[2, 1])
	@test isapprox(0.06590540043511113, C[1, 3])
	@test isapprox(0.06590540043511113, C[3, 1])
end

@testset "kernelmat implementation with different vectors" begin
	x1 = [8.1]
	x2 = [5, 6, 7, 8, 9, 10]
	theta = (2, 0.000001, [2])
	C = kernelmat(x1, x2, rbf_kernel, theta)
	@test size(C) == (1, 6)
	@test isapprox(0.6016359087145513, C[1, 1])
	@test isapprox(1.1524581473436002, C[1, 2])
	@test isapprox(1.7192655272050845, C[1, 3])
	@test isapprox(1.2736632287434861, C[1, 6])
end

@testset "kernelmat with duplicates" begin
	x1 = [6, 6, 8, 10]
	x2 = [6, 6, 8, 10]
	theta = (3, 0.000001, [2])
	C = kernelmat(x1, x2, rbf_kernel, theta)
	@test size(C) == (4, 4)
	@test isapprox(3.000001, C[1, 1])
	@test isapprox(3.000001, C[2, 2])
	@test isapprox(3.000001, C[2, 1])
	@test isapprox(3.000001, C[1, 2])
	@test isapprox(1.8195919791379003, C[1, 3])
	@test isapprox(1.8195919791379003, C[3, 2])
	@test isapprox(0.4060058497098381, C[1, 4])
end

@testset "train_gp implementation" begin
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta = (1, 2, [1])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)
	@test isapprox(gaussian_process.X, x)
	@test isapprox(gaussian_process.y, vec(y) .- mean(y))
	@test gaussian_process.kernel == rbf_kernel
	@test gaussian_process.theta == theta
	@test isapprox(
		gaussian_process.L[1:4],
		[
			1.7320508075688772,
			0.35018063965685026,
			2.151584212075994e-6,
			1.3219790295063264e-11,
		],
		atol = 1e-10,
	)
end

@testset "train_gp 2D implementation" begin
	X = [1.0 2.0; 6.0 8.0; 10.0 12.0]
	y = [1.0, 2.0, 3.0]
	theta = (1.2, 3, [3, 1.5])

	gaussian_process = train_gp(X, vec(y) .- mean(y), rbf_kernel, theta)
	@test isapprox(gaussian_process.X, X)
	@test isapprox(gaussian_process.y, vec(y) .- mean(y))
	@test gaussian_process.kernel == rbf_kernel
	@test gaussian_process.theta == theta
	@test isapprox(
		gaussian_process.L[1:3],
		[
			2.04939015319192,
			4.897945689810919e-5,
			1.452924160126954e-12,
		],
		atol = 1e-10,
	)
end

@testset "predict_gp implementation" begin
	x = [1, 2, 6, 8, 10]
	x_pred = [3, 5, 7, 9]
	y = sin.(x) + x
	theta = (2.4, 3.2, [2.1])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)

	pred_mean, variance = predict_gp(gaussian_process, x_pred)
	@test isapprox(
		pred_mean,
		[-1.5566430776763678, -0.2974091873260318, 1.1005084532599916, 2.029430601035582],
		atol = 1e-10,
	)
	@test isapprox(
		variance,
		[4.582923899392431, 4.669809804425466, 4.29505035071876, 4.294982961807127],
		atol = 1e-10,
	)
end

@testset "predict_gp implementation (prevent numerical imprecision due to using inv(C))" begin
	x = [0.3892003197157624, 0.5616257687746956, 0.5648610777316219, 0.5844942673941598, 0.8815602928365971]
	x_pred = [0.4834318626076195, 0.6357029565499659]
	y = [0.41054375048196934, 0.3479506821380999, 0.4410863311645655, 0.6355502145769517, 0.3857916988723882]
	theta = (0.6083464846949249, 8.491859326831275e-8, [0.5699792571106884])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)

	pred_mean, variance = predict_gp(gaussian_process, x_pred)
	@test isapprox(
		pred_mean,
		[-0.6188468428445049, 0.8061857176944613],
		atol = 1e-10,
	)
	@test isapprox(
		variance,
		[1.1528722019260584e-6, 1.5152706662746596e-6],
		atol = 1e-10,
	)
end

@testset "predict_gp 2D implementation" begin
	X = [1.0 2.0; 6.0 8.0; 10.0 12.0]
	X_pred = [3.0 4.0; 5.0 6.0]
	y = [1.0, 2.0, 3.0]
	theta = (1.2, 3, [3, 1.5])

	gaussian_process = train_gp(X, vec(y) .- mean(y), rbf_kernel, theta)

	pred_mean, variance = predict_gp(gaussian_process, X_pred)
	@test isapprox(
		pred_mean,
		[-0.0940716109207601, -0.0037015909741687143],
		atol = 1e-10,
	)
	@test isapprox(
		variance,
		[4.162742420852604, 4.148098647194509],
		atol = 1e-10,
	)
end

@testset "log_m_likelihood implementation" begin
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta = (1, 2, [1])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)
	@test isapprox(log_m_likelihood(gaussian_process), -14.456233482091134)
end

@testset "log_m_likelihood implementation (different values)" begin
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta = (1.4, 2.4, [1.2])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)
	@test isapprox(log_m_likelihood(gaussian_process), -13.296279329799068, atol = 1e-6)
end

# Make sure to implement the log_m_likelihood using the Cholesky decomposition instead of inv(C)
@testset "log_m_likelihood implementation (prevent numerical imprecision due to using inv(C))" begin
	x = [0.0714158148444275, 0.17324433300039244, 0.3747102727214584, 0.4643341716252696, 0.8745212987416857]
	y = [0.6132585539703432, 0.28123203644614425, 0.19297581560384625, 0.9059478236872945, 0.42316630445016234]
	theta = (0.7046148627175896, 3.5722581093189376e-7, [0.8309047770540186])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)
	@test isapprox(log_m_likelihood(gaussian_process), -47007.920846199064, atol = 1e-5)
end

@testset "grad_rbf implementation" begin
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta = (2, 2, [2])

	gp = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)

	@test isapprox(
		grad_rbf(x[1, :], gp.X[2, :], gp.theta),
		[0.8824969025845955, 0.0, 0.22062422564614886],
	)
	@test isapprox(
		grad_rbf(x[5, :], gp.X[4, :], gp.theta),
		[0.6065306597126334, 0.0, 0.6065306597126334],
	)
	@test isapprox(grad_rbf(x[3, :], gp.X[3, :], gp.theta), [1.0, 0, 0])
end

@testset "grad_logmlik implementation" begin
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta = (1, 2, [1])

	gaussian_process = train_gp(x, vec(y) .- mean(y), rbf_kernel, theta)

	@test isapprox(grad_logmlik(gaussian_process), [-2.020158404940758, -0.0, -1.182970693344678])
end

# The next testset is in case you want to implement gradient descent
# with status return value. However, if you want to debug your gradient
# descent in another way there is another test set further down
@testset "gradientdescent w. status" begin
	# Let us minimize the function x[1]^2 + x[2]^2 using gradient descent
	# the gradient ist given by the following function
	f(x) = x[1]^2 + x[2]^2
	f_grad(x) = [2 * x[1], 2 * x[2]]

	# First Check: moving in right direction
	x_star, x_list, status = gradientdescent(
		f_grad,
		[1, 20],
		[[0, 0] [100, 100]],
		stepsize = 1e-10,
		eps = 1e-1,
		max_iter = 2,
	)

	# After two tiny steps value of the function should be smaller than at initialization:
	@test f(x_star) < f([1, 20])

	# After some steps, x_star should be "relatively" close to 0 ;)
	x_star, x_list, status = gradientdescent(
		f_grad,
		[1, 20],
		[[0, 0] [100, 100]],
		stepsize = 1e-1,
		eps = 1e-1,
		max_iter = 50,
	)


	@test norm(x_star) < 1e-1
	@test status == 1

	# be careful about stepsize
	x_star, x_list, status = gradientdescent(
		f_grad,
		[1, 20],
		[[0, 0] [100, 100]],
		stepsize = 1e1,
		eps = 1e-1,
		max_iter = 2,
	)

	@test status == 10


	# be careful about stepsize
	x_star, x_list, status = gradientdescent(
		f_grad,
		[-1, -20],
		[[0, 0] [100, 100]],
		stepsize = 1e1,
		eps = 1e-1,
		max_iter = 2,
	)



	@test status == -10
end

# Gradient descent without status, uncomment if applicable.
# @testset "gradientdescent w/0 status" begin
#     # Let us minimize the function x[1]^2 + x[2]^2 using gradient descent
#     # the gradient ist given by the following function
#     f(x) = x[1]^2 + x[2]^2
#     f_grad(x) = [2*x[1], 2*x[2]]

#     ## First Check: moving in right direction
#     x_star = gradientdescent(f_grad,
#     [1, 20],
#     [[0, 0] [100, 100]],
#     stepsize = 1e-10,
#     eps = 1e-1,
#     max_iter = 2)

#     # After two tiny steps value of the function should be smaller than at initialization:
#     @test f(x_star) < f([1, 20])

#     # After some steps, x_star should be "relatively" close to 0 ;)
#     x_star = gradientdescent(f_grad,
#                     [1, 20],
#                     [[0, 0] [100, 100]],
#                     stepsize = 1e-1,
#                     eps = 1e-1,
#                     max_iter = 50)


#     @test norm(x_star) < 1e-1

# end

# test makes sure that your optimize_theta indeed increases the log m log_m_likelihood
@testset "optimize_theta implementation" begin
	#initialize data and theta
	x = [1, 2, 6, 8, 10]
	y = sin.(x) + x
	theta_init = (1, 1, [3])

	#train initial gp and compute log_m_likelihood
	gp_init = train_gp(x, y .- mean(y), rbf_kernel, theta_init)
	ll_init = log_m_likelihood(gp_init)

	#run optimize_theta for small number of iterations w. small stepsize
	#comment out status if not used and changed arguments of optimize_theta
	#function if applicable
	theta_star, thetas_list =
		optimize_theta(x, y .- mean(y), theta_init; stepsize = 1e-5, eps = 1e-3, max_iter = 10)

	#train gp w. found parameters
	gp_star = train_gp(x, y .- mean(y), rbf_kernel, theta_star)

	#compare both log_m_likelihoods
	ll_star = log_m_likelihood(gp_star)

	@test ll_init < ll_star
	@test theta_star[1:2] == (1.0004904253277878, 1.0) #test that first two dimensions did not change
	@test theta_star[3] < theta_init[3] #test that l-scale moved in right direction
end
