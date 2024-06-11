# Sample solution for the SEIR exercise
#
# 2024 by Ralf Herbrich
# Hasso-Plattner Institute

include("factors.jl")

module TrueSkill

export twoplayer, twoteams, threeplayer

using Plots
using LaTeXStrings
using ..Factors
using ..DistributionCollections
using ..GaussianDistribution


# Computes the TrueSkills for a two-player game
function twoplayer(skill1::Gaussian1D, skill2::Gaussian1D, beta)
	# Implement the algorithm for the two player setup of TrueSkill, as shown on slide 7 in unit 5.
	# Note that the match outcome is incorporated in the model implicitly (the "left" player, i.e., the one with the lower index wins).
	# Return the posterior distributions for skill 1 and skill 2, after applying the sum-product algorithm.
	db = DistributionBag(Gaussian1D(0, 0))

	# Create the graph structure and run the message passing algorithm.
	# Skill variables for the two players
	player_1_skill = GaussianFactor(db, add!(db), skill1)
	player_2_skill = GaussianFactor(db, add!(db), skill2)

	# Performance variables for the two players
	player_1_performance = GaussianMeanFactor(db, player_1_skill.x, add!(db), beta^2)
	player_2_performance = GaussianMeanFactor(db, player_2_skill.x, add!(db), beta^2)

	# Difference of performances
	performance_diff = WeightedSumFactor(db, player_1_performance.y, player_2_performance.y, add!(db), 1, -1)
	trunk = GreaterThanFactor(db, performance_diff.z)

	# Gaussian prior for the two player skills
	update_msg_to_x!(player_1_skill)
	update_msg_to_x!(player_2_skill)

	# Gaussian likelihood of performance for the two players
	update_msg_to_y!(player_1_performance)
	update_msg_to_y!(player_2_performance)

	# Difference factor for the player performances
	update_msg_to_z!(performance_diff)

	# Match outcome factor
	update_msg_to_x!(trunk)

	# run the message passing schedule
	update_msg_to_x!(performance_diff)
	update_msg_to_y!(performance_diff)

	update_msg_to_x!(player_1_performance)
	update_msg_to_x!(player_2_performance)

	# return the marginals of the skill values of all involved players
	return db.bag[player_1_skill.x], db.bag[player_2_skill.x]
end

function twoteams(
	skill11::Gaussian1D,
	skill12::Gaussian1D,
	skill21::Gaussian1D,
	skill22::Gaussian1D,
	β,
)
	# Implement the TrueSkill algorithm for a match between two teams of two players each,
	# with team 1 consisting of player 11 and 12 and team 2 consisting of player 21 and 22.
	# Note that the match outcome is incorporated in the model implicitly (the "left" team, i.e., the one with the lower index wins).
	# Return the posterior distributions for the four skills (in the same order as in the function header),
	# after applying the sum-product algorithm.
	db = DistributionBag(Gaussian1D(0, 0))

	# Create the graph structure and run the message passing algorithm.
	# Skill variables for the four players
	player_11_skill = GaussianFactor(db, add!(db), skill11)
	player_12_skill = GaussianFactor(db, add!(db), skill12)
	player_21_skill = GaussianFactor(db, add!(db), skill21)
	player_22_skill = GaussianFactor(db, add!(db), skill22)

	# Performance variables for the four players
	player_11_performance = GaussianMeanFactor(db, player_11_skill.x, add!(db), β^2)
	player_12_performance = GaussianMeanFactor(db, player_12_skill.x, add!(db), β^2)
	player_21_performance = GaussianMeanFactor(db, player_21_skill.x, add!(db), β^2)
	player_22_performance = GaussianMeanFactor(db, player_22_skill.x, add!(db), β^2)

	# Performance for the two team performances
	team_1_performance = WeightedSumFactor(db, player_11_performance.y, player_12_performance.y, add!(db), 1, 1)
	team_2_performance = WeightedSumFactor(db, player_21_performance.y, player_22_performance.y, add!(db), 1, 1)

	# Difference of team performances
	performance_diff = WeightedSumFactor(db, team_1_performance.z, team_2_performance.z, add!(db), 1, -1)
	trunk = GreaterThanFactor(db, performance_diff.z)

	# Gaussian prior for the four player skills
	update_msg_to_x!(player_11_skill)
	update_msg_to_x!(player_12_skill)
	update_msg_to_x!(player_21_skill)
	update_msg_to_x!(player_22_skill)

	# Gaussian likelihood of performance for the four players
	update_msg_to_y!(player_11_performance)
	update_msg_to_y!(player_12_performance)
	update_msg_to_y!(player_21_performance)
	update_msg_to_y!(player_22_performance)

	# Weighted sum factor for the team performances
	update_msg_to_z!(team_1_performance)
	update_msg_to_z!(team_2_performance)

	# Difference factor for the team performances
	update_msg_to_z!(performance_diff)

	# Match outcome factor
	update_msg_to_x!(trunk)

	# run the message passing schedule
	update_msg_to_x!(performance_diff)
	update_msg_to_y!(performance_diff)

	update_msg_to_x!(team_1_performance)
	update_msg_to_y!(team_1_performance)
	update_msg_to_x!(team_2_performance)
	update_msg_to_y!(team_2_performance)

	update_msg_to_x!(player_11_performance)
	update_msg_to_x!(player_12_performance)
	update_msg_to_x!(player_21_performance)
	update_msg_to_x!(player_22_performance)

	# return the marginals of the skill values of all involved players
	return db.bag[player_11_skill.x], db.bag[player_12_skill.x], db.bag[player_21_skill.x], db.bag[player_22_skill.x]
end

# Computes the TrueSkills for a three-team player game
function threeplayer(skill1::Gaussian1D, skill2::Gaussian1D, skill3::Gaussian1D, β, min_diff = 1e-6)
	# Implement the TrueSkill algorithm for a three players, with matches between player 1 and 2 and between player 2 and 3.
	# Note that the match outcome is incorporated in the model implicitly (the "left" player, i.e., the one with the lower index wins).
	# Return the posterior distributions for the three skills (in the same order as in the function header),
	# after applying the sum-product algorithm.
	# This example needs iteration to converge. Iteration should occur until the highest measured abs_diff in one
	# iteration is lower than the provided value of min_diff.
	db = DistributionBag(Gaussian1D(0, 0))

	# Create the graph structure and run the message passing algorithm.
	# Skill variables for the three players
	player_1_skill = GaussianFactor(db, add!(db), skill1)
	player_2_skill = GaussianFactor(db, add!(db), skill2)
	player_3_skill = GaussianFactor(db, add!(db), skill3)

	# Performance variables for the three players
	player_1_performance = GaussianMeanFactor(db, player_1_skill.x, add!(db), β^2)
	player_2_performance = GaussianMeanFactor(db, player_2_skill.x, add!(db), β^2)
	player_3_performance = GaussianMeanFactor(db, player_3_skill.x, add!(db), β^2)

	# Difference of pairwise player performances
	performance_diff_12 = WeightedSumFactor(db, player_1_performance.y, player_2_performance.y, add!(db), 1, -1)
	performance_diff_23 = WeightedSumFactor(db, player_2_performance.y, player_3_performance.y, add!(db), 1, -1)
	trunk_12 = GreaterThanFactor(db, performance_diff_12.z)
	trunk_23 = GreaterThanFactor(db, performance_diff_23.z)

	# Gaussian prior for the three player skills
	update_msg_to_x!(player_1_skill)
	update_msg_to_x!(player_2_skill)
	update_msg_to_x!(player_3_skill)

	# Gaussian likelihood of performance for the three players
	update_msg_to_y!(player_1_performance)
	update_msg_to_y!(player_2_performance)
	update_msg_to_y!(player_3_performance)

	# Continuously apply the message passing algorithm until convergence, where the threshold is determined by the maximum difference.
	# If we'd choose the minimum difference, the algorithm might terminate prematurely, as the first performance difference might be zero.
	δ_max = Inf
	while δ_max > min_diff
		δ_max = max(
			# Difference factors for the player performances and match outcome factors
			update_msg_to_z!(performance_diff_12),
			update_msg_to_x!(trunk_12),
			update_msg_to_y!(performance_diff_12),
			update_msg_to_z!(performance_diff_23),
			update_msg_to_x!(trunk_23),
			update_msg_to_x!(performance_diff_23),
		)
	end

	# run the message passing schedule
	update_msg_to_x!(performance_diff_12)
	update_msg_to_y!(performance_diff_23)

	update_msg_to_x!(player_1_performance)
	update_msg_to_x!(player_2_performance)
	update_msg_to_x!(player_3_performance)

	# return the marginals of the skill values of all involved players
	return db.bag[player_1_skill.x], db.bag[player_2_skill.x], db.bag[player_3_skill.x]
end

end
