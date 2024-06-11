# A set of types and functions for factors over gaussian variables
#
# Hasso-Plattner Institute

include("gaussian.jl")
include("distributionbag.jl")
module Factors
using Distributions: Distributions
export GaussianFactor, GaussianMeanFactor, WeightedSumFactor, GreaterThanFactor, update_msg_to_x!, update_msg_to_y!, update_msg_to_z!

using ..DistributionCollections
using ..GaussianDistribution

abstract type Factor end


struct GaussianFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	prior::Gaussian1D
	msg_to_x::Int64
end

GaussianFactor(db::DistributionBag{Gaussian1D}, x::Int64, prior) = GaussianFactor(db, x, prior, add!(db))

function update_msg_to_x!(f::GaussianFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	old_marginal = f.db.bag[f.x]

	# Update the distributions for the marginal of x and the message from this factor to x.
	f.db.bag[f.msg_to_x] = f.prior
	f.db.bag[f.x] = msg_to_factor_from_x * f.db.bag[f.msg_to_x]

	# Return the absolute difference of the old and the new marginal of x.
	return absdiff(old_marginal, f.db.bag[f.x])
end

struct GaussianMeanFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	beta_squared::Float64
	msg_to_x::Int64
	msg_to_y::Int64
end

GaussianMeanFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, beta_squared) = GaussianMeanFactor(db, x, y, beta_squared, add!(db), add!(db))

function update_msg_to_x!(f::GaussianMeanFactor)
	msg_to_factor_from_y = f.db.bag[f.y] / f.db.bag[f.msg_to_y]
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	old_marginal = f.db.bag[f.x]

	# Update the distributions for the marginal of x and the message from this factor to x.
	# f.db.bag[f.msg_to_x] = Gaussian1DFromMeanVariance(mean(msg_to_factor_from_y), variance(msg_to_factor_from_y) + f.beta_squared)
	f.db.bag[f.msg_to_x] = Gaussian1D(msg_to_factor_from_y.tau / (1 + msg_to_factor_from_y.rho * f.beta_squared), msg_to_factor_from_y.rho / (1 + msg_to_factor_from_y.rho * f.beta_squared))
	f.db.bag[f.x] = msg_to_factor_from_x * f.db.bag[f.msg_to_x]

	# Return the absolute difference of the old and the new marginal of x.
	return absdiff(old_marginal, f.db.bag[f.x])
end

function update_msg_to_y!(f::GaussianMeanFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	msg_to_factor_from_y = f.db.bag[f.y] / f.db.bag[f.msg_to_y]
	old_marginal = f.db.bag[f.y]

	# Update the distributions for the marginal of y and the message from this factor to y.
	# f.db.bag[f.msg_to_y] = Gaussian1DFromMeanVariance(isnan(mean(msg_to_factor_from_x)) ? 0.0 : mean(msg_to_factor_from_x),  variance(msg_to_factor_from_x) + f.beta_squared)
	f.db.bag[f.msg_to_y] = Gaussian1D(msg_to_factor_from_x.tau / (1 + msg_to_factor_from_x.rho * f.beta_squared), msg_to_factor_from_x.rho / (1 + msg_to_factor_from_x.rho * f.beta_squared))
	f.db.bag[f.y] = msg_to_factor_from_y * f.db.bag[f.msg_to_y]

	# Return the absolute difference of the old and the new marginal of y.
	return absdiff(old_marginal, f.db.bag[f.y])
end


struct WeightedSumFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	z::Int64
	a::Float64
	b::Float64
	msg_to_x::Int64
	msg_to_y::Int64
	msg_to_z::Int64
end

WeightedSumFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, a, b) = WeightedSumFactor(db, x, y, z, a, b, add!(db), add!(db), add!(db))

function update_msg_to_x!(f::WeightedSumFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	msg_to_factor_from_y = f.db.bag[f.y] / f.db.bag[f.msg_to_y]
	msg_to_factor_from_z = f.db.bag[f.z] / f.db.bag[f.msg_to_z]
	old_marginal = f.db.bag[f.x]

	# Update the distributions for the marginal of x and the message from this factor to x.
	f.db.bag[f.msg_to_x] = Gaussian1DFromMeanVariance((mean(msg_to_factor_from_z) - f.b * mean(msg_to_factor_from_y)) / f.a, (variance(msg_to_factor_from_z) + f.b^2 * variance(msg_to_factor_from_y)) / f.a^2)
	f.db.bag[f.x] = msg_to_factor_from_x * f.db.bag[f.msg_to_x]

	# Return the absolute difference of the old and the new marginal of x.
	return absdiff(old_marginal, f.db.bag[f.x])
end

function update_msg_to_y!(f::WeightedSumFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	msg_to_factor_from_y = f.db.bag[f.y] / f.db.bag[f.msg_to_y]
	msg_to_factor_from_z = f.db.bag[f.z] / f.db.bag[f.msg_to_z]
	old_marginal = f.db.bag[f.y]

	# Update the distributions for the marginal of y and the message from this factor to y.
	f.db.bag[f.msg_to_y] = Gaussian1DFromMeanVariance((mean(msg_to_factor_from_z) - f.a * mean(msg_to_factor_from_x)) / f.b, (variance(msg_to_factor_from_z) + f.a^2 * variance(msg_to_factor_from_x)) / f.b^2)
	f.db.bag[f.y] = msg_to_factor_from_y * f.db.bag[f.msg_to_y]

	# Return the absolute difference of the old and the new marginal of y.
	return absdiff(old_marginal, f.db.bag[f.y])
end

function update_msg_to_z!(f::WeightedSumFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	msg_to_factor_from_y = f.db.bag[f.y] / f.db.bag[f.msg_to_y]
	msg_to_factor_from_z = f.db.bag[f.z] / f.db.bag[f.msg_to_z]
	old_marginal = f.db.bag[f.z]

	# Update the distributions for the marginal of z and the message from this factor to z.
	f.db.bag[f.msg_to_z] = Gaussian1DFromMeanVariance(f.a * mean(msg_to_factor_from_x) + f.b * mean(msg_to_factor_from_y), f.a^2 * variance(msg_to_factor_from_x) + f.b^2 * variance(msg_to_factor_from_y))
	f.db.bag[f.z] = msg_to_factor_from_z * f.db.bag[f.msg_to_z]

	# Return the absolute difference of the old and the new marginal of z.
	return absdiff(old_marginal, f.db.bag[f.z])
end

struct GreaterThanFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	eps::Float64
	msg_to_x::Int64
end

# Initializes the greater than factor. Eps is the value that should be used for comparison, it is set to 0.0 at all times for now.
GreaterThanFactor(db::DistributionBag{Gaussian1D}, x::Int64, eps = 0.0) = GreaterThanFactor(db, x, eps, add!(db))

function v(t::Float64, ε::Float64 = 0.0)
	standardGaussian = Distributions.Normal(0, 1)
	return Distributions.cdf(standardGaussian, t - ε) < floatmin(Float64) ? 0.0 : Distributions.pdf(standardGaussian, t - ε) / Distributions.cdf(standardGaussian, t - ε)
end

function w(t::Float64)
	return v(t) * (v(t) + t)
end

function update_msg_to_x!(f::GreaterThanFactor)
	msg_to_factor_from_x = f.db.bag[f.x] / f.db.bag[f.msg_to_x]
	old_marginal = f.db.bag[f.x]

	# Update the distributions for the marginal of x and the message from this factor to x.
	# (See unit 5, slide 16 for the truncated Gaussian. You can use the Distributions package to get the PDF and CDF or a normal Distribution.)
	μ̂ = mean(msg_to_factor_from_x) + sqrt(variance(msg_to_factor_from_x)) * v(mean(msg_to_factor_from_x) / sqrt(variance(msg_to_factor_from_x)))
	σ̂² = variance(msg_to_factor_from_x) * (1 - w(mean(msg_to_factor_from_x) / sqrt(variance(msg_to_factor_from_x))))
	p̂ = Gaussian1DFromMeanVariance(μ̂, σ̂²)
	f.db.bag[f.msg_to_x] = p̂ / msg_to_factor_from_x
	f.db.bag[f.x] = msg_to_factor_from_x * f.db.bag[f.msg_to_x]

	# Return the absolute difference of the old and the new marginal of x.
	return absdiff(old_marginal, f.db.bag[f.x])
end

end
