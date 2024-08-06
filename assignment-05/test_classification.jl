using Test, Random
include("gaussian.jl")
include("bayesian_linear_classification.jl")
using .GaussianDistribution: GaussianND
using ..Classification

@testset "Bayesian Linear Classification Tests 5x2" begin

	expected_posterior = GaussianND([-0.15953550310002831, -0.1052044292243254], [0.7026006854114938 -0.32208254318353596; -0.32208254318353596 0.5087814127023217])
	expected_predictions = [0.4421423043148064, 0.4453507997899878, 0.4178668924313136, 0.49357665278427687, 0.43707208708601075]

	Random.seed!(1234)
	phi = rand(5, 2)
	y = vcat(ones(2), -1.0 * ones(3))

	posterior = learn_weight_distribution(phi, y)
	predictions = prediction(phi, posterior)

	@test size(posterior.mean) == (2,)
	@test size(posterior.cov) == (2, 2)
	@test size(predictions) == (5,)

	@test posterior.mean ≈ expected_posterior.mean atol = 1e-5
	@test posterior.cov ≈ expected_posterior.cov atol = 1e-5
	@test expected_predictions ≈ predictions atol = 1e-5
end

@testset "Bayesian Linear Classification Tests 10x3" begin

	expected_posterior = GaussianND(
		[-0.3699743540948342, -0.008246797243952858, 0.13000374705343432],
		[0.5555301428560878 -0.27076508568341434 -0.21235525291853316; -0.27076508568341434 0.48759396790472076 -0.10709417169186232; -0.2123552529185331 -0.10709417169186232 0.5135130790754637],
	)
	expected_predictions = [0.443205840700872, 0.4443548811518401, 0.41761430442822023, 0.5290467926635699, 0.43390861108792234, 0.4581721867975401, 0.41371779076639703, 0.4090181652838063, 0.4183420190728624, 0.4116381706473018]

	Random.seed!(1234)
	phi = rand(10, 3)
	y = vcat(ones(5), -1.0 * ones(5))

	posterior = learn_weight_distribution(phi, y)
	predictions = prediction(phi, posterior)

	@test size(posterior.mean) == (3,)
	@test size(posterior.cov) == (3, 3)
	@test size(predictions) == (10,)

	@test posterior.mean ≈ expected_posterior.mean atol = 1e-5
	@test posterior.cov ≈ expected_posterior.cov atol = 1e-5
	@test expected_predictions ≈ predictions atol = 1e-5
end

@testset "Bayesian Linear Classification Tests 10x3 Correct Use of Beta and Tau" begin

	expected_posterior = GaussianND(
		[-0.7399487081896684, -0.016493594487905716, 0.26000749410686863],
		[2.2221205714243513 -1.0830603427336574 -0.8494210116741326; -1.0830603427336574 1.950375871618883 -0.4283766867674493; -0.8494210116741324 -0.4283766867674493 2.054052316301855],
	)
	expected_predictions = [0.443205840700872, 0.4443548811518401, 0.41761430442822023, 0.5290467926635699, 0.43390861108792234, 0.4581721867975401, 0.41371779076639703, 0.4090181652838063, 0.4183420190728624, 0.4116381706473018]

	Random.seed!(1234)
	phi = rand(10, 3)
	y = vcat(ones(5), -1.0 * ones(5))

	posterior = learn_weight_distribution(phi, y; beta = 2.0, tau = 2.0)
	predictions = prediction(phi, posterior; beta = 2.0)

	@test size(posterior.mean) == (3,)
	@test size(posterior.cov) == (3, 3)
	@test size(predictions) == (10,)

	@test posterior.mean ≈ expected_posterior.mean atol = 1e-5
	@test posterior.cov ≈ expected_posterior.cov atol = 1e-5
	@test expected_predictions ≈ predictions atol = 1e-5
end


# This test set is just to help you debugging, use it if you're stuck.

@testset "Bayesian Linear Classification Optional Debugging Helper Tests" begin
	Random.seed!(1234)

	phi_i = rand(3)
	y_i = 1.0
	beta = 1.0
	mu = rand(size(phi_i, 1))
	a_i = rand()
	b_i = rand()
	sigma = rand(size(mu, 1), size(mu, 1))

	z = get_z(phi_i, sigma)
	e = get_e(phi_i, z)
	f = get_f(phi_i, mu)
	c = get_c(b_i, e)

	m = get_m(y_i, a_i, f, e, c)
	s_squared = get_s_squared(c, e, beta)

	s = sqrt(s_squared)
	V = v(m / s, 0.0)
	W = w(m / s, 0.0)

	tau = get_tau(s_squared, m, V, W)
	rho = get_rho(W, s_squared)

	a_new = get_a_new(tau, rho, beta)
	b_new = get_b_new(rho, beta)

	delta = maximum([abs(a_i - a_new), abs(b_i - b_new)])

	d = get_d(b_new, b_i, e)

	sigma = get_new_sigma(sigma, b_new, b_i, d, z)
	mu = get_new_mu(mu, y_i, a_new, a_i, b_new, b_i, f, d, z)

	@test z ≈ [0.22246325462110927, 0.25975983298623184, 0.9451074482763887] atol = 1e-5
	@test e ≈ 0.4217271510670675 atol = 1e-5
	@test f ≈ 0.5715585109976284 atol = 1e-5
	@test c ≈ 0.6644962522760809 atol = 1e-5
	@test m ≈ 0.25523090121276865 atol = 1e-5
	@test s_squared ≈ 1.6346569293995818 atol = 1e-5
	@test V ≈ 0.6752933616849291 atol = 1e-5
	@test W ≈ 0.5908281183035293 atol = 1e-5
	@test tau ≈ 1.5162990063268187 atol = 1e-5
	@test rho ≈ 0.8833417446272849 atol = 1e-5
	@test a_new ≈ 0.8051109208684246 atol = 1e-5
	@test b_new ≈ 0.46902892008168123 atol = 1e-5
	@test delta ≈ 0.3265180274530382 atol = 1e-5
	@test sigma ≈ [0.22390752374644068 0.09587734621148161 0.605519493369594; 0.5490653940395602 0.13636701176271981 0.21678450629336937; 1.030775815526946 0.912980839365017 1.1837444373020225] atol = 1e-5
	@test mu ≈ [0.9042064878613684, 0.36474270319205904, 0.4365736895755471] atol = 1e-5
end
