include("factors.jl")

using Test
using ..Factors
using ..DistributionCollections
using ..GaussianDistribution

# Test GaussianFactor
function test_gaussian_factor()
    # Test update!
    @testset "GaussianFactor - msg_to_x!" begin
        bag = DistributionBag(Gaussian1D(0, 0))
        s = add!(bag)
        factor = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(2, 42))
        factor2 = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(1, 1))

        @test factor.msg_to_x == 2
    end

    # Test update!
    @testset "GaussianFactor - update!" begin
        bag = DistributionBag(Gaussian1D(0, 0))
        s = add!(bag)
        factor = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(2, 42))
        factor2 = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(1, 1))

        @test bag[s] == Gaussian1D(0, 0)
        @test update_msg_to_x!(factor) ≈ 0.1543033499620919 atol = 1e-6
        @test update_msg_to_x!(factor) ≈ 0.0
        @test mean(bag[s]) ≈ 2.0 atol = 1e-6
        @test variance(bag[s]) ≈ 42.0 atol = 1e-6
        @test_throws MethodError update_msg_to_y!(factor)
        @test update_msg_to_x!(factor2) ≈ 1.0 atol = 1e-6
        @test mean(bag[s]) ≈ 1.0232558139534884 atol = 1e-6
        @test variance(bag[s]) ≈ 0.9767441860465117 atol = 1e-6
        @test_throws MethodError update_msg_to_y!(factor)
    end

end

# Test GaussianMeanFactor
function test_gaussian_mean_factor()
    @testset "GaussianMeanFactor - msg_to_x, msg_to_y" begin
        bag = DistributionBag(Gaussian1D(0, 0))
        s1 = add!(bag)
        s2 = add!(bag)
        f1 = GaussianFactor(bag, s1, Gaussian1DFromMeanVariance(3.0, 1.0))
        factor = GaussianMeanFactor(bag, s2, s1, 0.5)

        @test factor.msg_to_x == 4
        @test factor.msg_to_y == 5
    end

    # Test update!
    @testset "GaussianMeanFactor - update!" begin
        bag = DistributionBag(Gaussian1D(0, 0))
        s1 = add!(bag)
        s2 = add!(bag)
        f1 = GaussianFactor(bag, s1, Gaussian1DFromMeanVariance(3.0, 1.0))
        factor = GaussianMeanFactor(bag, s2, s1, 0.5)

        @test update_msg_to_x!(f1) ≈ 3.0
        @test update_msg_to_x!(factor) ≈ 2.0
        @test mean(bag[s1]) ≈ 3.0
        @test mean(bag[s2]) ≈ 3.0
        @test variance(bag[s1]) ≈ 1.0
        @test variance(bag[s2]) ≈ 1.5

        @test update_msg_to_y!(factor) ≈ 0.0

        @test_throws MethodError update_msg_to_z!(factor)
    end

end

# Test WeightedSumFactor
function test_weighted_sum_factor()
    @testset "WeightedSumFactor - msg_to_x, msg_to_y, msg_to_z" begin
        bag = DistributionBag(Gaussian1D(0.0, 0.0))
        s1 = add!(bag)
        s2 = add!(bag)
        s3 = add!(bag)
        f1 = GaussianFactor(bag, s1, Gaussian1DFromMeanVariance(1, 1))
        f2 = GaussianFactor(bag, s2, Gaussian1DFromMeanVariance(2, 4))
        f3 = GaussianFactor(bag, s3, Gaussian1DFromMeanVariance(2, 0.5))
        factor = WeightedSumFactor(bag, s1, s2, s3, 0.5, 0.5)

        @test factor.msg_to_x == 7
        @test factor.msg_to_y == 8
        @test factor.msg_to_z == 9
    end

    # Test update!
    @testset "WeightedSumFactor - update!" begin
        bag = DistributionBag(Gaussian1D(0.0, 0.0))
        s1 = add!(bag)
        s2 = add!(bag)
        s3 = add!(bag)
        f1 = GaussianFactor(bag, s1, Gaussian1DFromMeanVariance(1, 1))
        f2 = GaussianFactor(bag, s2, Gaussian1DFromMeanVariance(2, 4))
        f3 = GaussianFactor(bag, s3, Gaussian1DFromMeanVariance(2, 0.5))
        factor = WeightedSumFactor(bag, s1, s2, s3, 0.5, 0.5)

        @test update_msg_to_x!(f1) ≈ 1.0
        @test update_msg_to_x!(f2) ≈ 0.5
        @test update_msg_to_z!(factor) ≈ 1.2
        @test mean(bag[s3]) ≈ 1.5 atol = 1e-6
        @test variance(bag[s3]) ≈ 1.25 atol = 1e-6

        @test update_msg_to_x!(f3) ≈ 4.0
        @test mean(bag[s3]) ≈ 1.8571428571428574 atol = 1e-6
        @test variance(bag[s3]) ≈ 0.35714285714285715 atol = 1e-6

        @test mean(bag[s1]) ≈ 1.0 atol = 1e-6
        @test variance(bag[s1]) ≈ 1.0 atol = 1e-6
        @test update_msg_to_x!(factor) ≈ 0.40824829046386313 atol = 1e-6
        @test mean(bag[s1]) ≈ 1.142857142857143 atol = 1e-6
        @test variance(bag[s1]) ≈ 0.8571428571428571 atol = 1e-6

        @test mean(bag[s2]) ≈ 2.0 atol = 1e-6
        @test variance(bag[s2]) ≈ 4.0 atol = 1e-6
        @test update_msg_to_y!(factor) ≈ 1.0000000000000002 atol = 1e-6
        @test mean(bag[s2]) ≈ 2.571428571428572 atol = 1e-6
        @test variance(bag[s2]) ≈ 1.7142857142857144 atol = 1e-6

        # @test_throws ArgumentError factor.update!(4)
    end
end

# Test GreaterThanFactor
function test_greater_than_factor()
    @testset "GreaterThanFactor - msg_to_x" begin
        bag = DistributionBag(Gaussian1D(0.0, 0.0))
        s = add!(bag)
        f1 = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(1, 1))
        factor = GreaterThanFactor(bag, s, 0.0)

        @test factor.msg_to_x == 3
    end

    # Test update!
    @testset "GreaterThanFactor - update!" begin
        bag = DistributionBag(Gaussian1D(0.0, 0.0))
        s = add!(bag)
        f1 = GaussianFactor(bag, s, Gaussian1DFromMeanVariance(1, 1))
        factor = GreaterThanFactor(bag, s, 0.0)

        @test update_msg_to_x!(f1) ≈ 1.0

        @test mean(bag[s]) ≈ 1.0
        @test variance(bag[s]) ≈ 1.0
        @test update_msg_to_x!(factor) ≈ 1.0448277182202785 atol = 1e-6
        @test mean(bag[s]) ≈ 1.2875999709391783 atol = 1e-6
        @test variance(bag[s]) ≈ 0.6296862857766055 atol = 1e-6

        @test_throws MethodError update_msg_to_y!(factor)
    end
end

# Run the tests
@testset "Factor functions tests" begin
    test_gaussian_factor()
    test_gaussian_mean_factor()
    test_weighted_sum_factor()
    test_greater_than_factor()
end
