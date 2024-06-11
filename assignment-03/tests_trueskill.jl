
include("trueskill.jl")

using .TrueSkill
using .GaussianDistribution
using Test

# Test the twoplayer function
@testset "twoplayer function" begin
    skill = Gaussian1DFromMeanVariance(25.0, 25.0 * 25.0 / (3.0 * 3.0))
    β = 25.0 / 6.0

    # Compute the posterior skills
    (s1, s2) = twoplayer(skill, skill, β)

    @test mean(s1) ≈ 29.205220870033607
    @test sqrt(variance(s1)) ≈ 7.194481348831082 atol = 1e-6
    @test mean(s2) ≈ 20.794779129966404 atol = 1e-6
    @test sqrt(variance(s2)) ≈ 7.194481348831082 atol = 1e-6
end

# Test the twoteams function
@testset "twoteams function" begin
    skill = Gaussian1DFromMeanVariance(25.0, 25.0 * 25.0 / (3.0 * 3.0))
    β = 25.0 / 6.0

    # Compute the posterior skills
    (s11, s12, s21, s22) = twoteams(skill, skill, skill, skill, β)

    @test mean(s11) ≈ mean(s12)
    @test mean(s12) ≈ 27.973540193587954 atol = 1e-6
    @test sqrt(variance(s11)) ≈ sqrt(variance(s11))
    @test sqrt(variance(s11)) ≈ 7.784760957252405 atol = 1e-6
    @test mean(s21) ≈ mean(s22)
    @test mean(s22) ≈ 22.026459806412053 atol = 1e-6
    @test sqrt(variance(s21)) ≈ sqrt(variance(s22))
    @test sqrt(variance(s22)) ≈ 7.784760957252405 atol = 1e-6
end

# Test the threeplayer function
@testset "threeplayer function" begin
    skill = Gaussian1DFromMeanVariance(25.0, 25.0 * 25.0 / (3.0 * 3.0))
    β = 25.0 / 6.0

    # Compute the posterior skills
    (s1, s2, s3) = threeplayer(skill, skill, skill, β)

    @test mean(s1) ≈ 31.311357968443478 atol = 1e-6
    @test sqrt(variance(s1)) ≈ 6.698818832900481 atol = 1e-6
    @test mean(s2) ≈ 24.999999999999986 atol = 1e-6
    @test sqrt(variance(s2)) ≈ 6.2384699253598885 atol = 1e-6
    @test mean(s3) ≈ 18.688642031556533 atol = 1e-6
    @test sqrt(variance(s3)) ≈ 6.698818832900474 atol = 1e-6
end
