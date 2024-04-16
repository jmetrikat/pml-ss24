include("discrete.jl")

using .DiscreteDistribution
using Test

@testset "Discrete constructor only specified by length" begin
    d = Discrete(3)
    @test all(d.logP == [0.3333333333333333, 0.3333333333333333, 0.3333333333333333])
end

@testset "Discrete constructor specified element wise" begin
    logProbs = [0.0, 0.0]
    d = Discrete(logProbs)
    @test all(d.logP == logProbs)
end

@testset "Multiplication of two discrete distributions" begin
    d = Discrete([0.0, 2.0, -1.0]) * Discrete([1.0, 0.0, 1.0])
    @test all(d.logP == [0.24472847105479764, 0.6652409557748219, 0.09003057317038046])
end

@testset "Division of two discrete distributions" begin
    d = Discrete([1.0, 2.0]) / Discrete([1.0, 0.0])
    @test d.logP == [0.11920292202211755, 0.8807970779778824]
end

@testset "Test logsumexp" begin
    d = logsumexp([1.0, -1.0])
    @test d == 1.1269280110429727
end

@testset "Conversion of logprobs to probabilities" begin
    d = ℙ(Discrete([1.0, -1.0]))
    @test all(d == [0.8807970779778823, 0.11920292202211753])
end
