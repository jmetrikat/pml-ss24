# A framework of the SEIR model
#
# 2024 by Ralf Herbrich
# Hasso-Plattner Institute

include("factors.jl")
module Seir
export seir_single_time_series, display_time_series


using Plots
using LaTeXStrings
using Printf
using ..Factors
using ..DistributionCollections
using ..DiscreteDistribution


"""
		seir_single_time_series
Should perform the computation presented in the assignment document.
Has 3 parameters:
	- steps: The number of time-steps to be taken into account
	- prior_probs: The prior probabilities to start the computation with
	- transition_probs: The transition probabilities

It should return the marginal distributions for each timestep as an array of discrete distributions.
"""
function seir_single_time_series(;
	steps = 100,
	prior_probs = [1.0, 1e-6, 1e-6, 1e-6],
	transition_probs = [0.95 0.05 0.0 0.0; 0.0 0.8 0.2 0.0; 0.0 0.0 0.7 0.3; 0.0 0.0 0.0 1.0]',
)
	"""
		Time series calculation for the SEIR model through Factor Graphs.
	"""
	# create a distribution bag to store the distributions over time
	db = DistributionBag(Discrete(4))
	marginal_distributions::Vector{Discrete{4}} = Vector{Discrete{4}}(undef, steps)

	# start with message from the prior factor and store the updated marginal as first timestep result
	x = add!(db)
	prior = Discrete(log.(prior_probs))
	f1 = PriorDiscreteFactor(db, x, prior)
	update_msg_to_x!(f1)
	marginal_distributions[1] = db[f1.x]

	# continue the forward pass with messages from the coupling factors and store the updated marginals as results
	for i in 2:steps
		y = add!(db)
		f2 = CouplingDiscreteFactor(db, x, y, Matrix(transition_probs))
		update_msg_to_y!(f2)
		marginal_distributions[i] = db[f2.y]
		x = y
	end

	return marginal_distributions


	"""
		Time series calculation for the SEIR model through Markov Chains.
	"""
	db = DistributionBag(Discrete(4))
	add!(db)
	db[1] = Discrete(prior_probs)

	# instead of using factor graph message passing, simply calculate the next state based on the previous one and the transition matrix
	for i in 2:steps
		d = Discrete(4)
		d.logP[1] = db.bag[i-1].logP[1] * transition_probs[1, 1]
		d.logP[2] = db.bag[i-1].logP[1] * transition_probs[2, 1] + db.bag[i-1].logP[2] * transition_probs[2, 2]
		d.logP[3] = db.bag[i-1].logP[2] * transition_probs[3, 2] + db.bag[i-1].logP[3] * transition_probs[3, 3]
		d.logP[4] = db.bag[i-1].logP[3] * transition_probs[4, 3] + db.bag[i-1].logP[4] * transition_probs[4, 4]
		add!(db)
		db[i] = d
	end

	return db.bag
end

"""
		display_time_series

Given an array of distributions, it should display the 4 probabilities along
the time axis. (Hint: This should call plot/plot! 4 times ;)).

It should display the generated plot.
"""
function display_time_series(ps::Vector{Discrete{T}}) where {T}
	p = plot(
		xlabel = L"t",
		ylabel = L"P(\mathrm{SEIR})",
		xtickfontsize = 14,
		ytickfontsize = 14,
		legendfontsize = 14,
		xguidefontsize = 16,
		yguidefontsize = 16,
		legend = :bottomright)

    # transform the log probabilities to probabilities for each time step
	susceptible = [ℙ(ps[i])[1] for i in 1:length(ps)]
	exposed = [ℙ(ps[i])[2] for i in 1:length(ps)]
	infected = [ℙ(ps[i])[3] for i in 1:length(ps)]
	recovered = [ℙ(ps[i])[4] for i in 1:length(ps)]

	t_steps = 1:length(ps)

    # plot the 4 distributions over 100 time steps
	plot!(p, t_steps, susceptible, label = "Susceptible")
	plot!(p, t_steps, exposed, label = "Exposed")
	plot!(p, t_steps, infected, label = "Infected")
	plot!(p, t_steps, recovered, label = "Recovered")

	display(p)
	savefig(p, @sprintf("seir-%.2f.png", susceptible[2]))
end

end
