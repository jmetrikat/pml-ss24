include("cholesky.jl")
using .Cholesky: cholesky_decomposition
using Test

A = [8.0 5.0 2.0; 5.0 5.0 0.0; 2.0 0.0 8.0]
L2 = [2.828 0.0 0.0; 1.768 1.369 0.0; 0.707 -0.913 2.582]
@testset "Cholesky decomposition test according to the exercise" begin
	L = cholesky_decomposition(A)
	@test isapprox(A, L * transpose(L))
	@test isapprox(L, L2; atol = 1e-3)
end

A2 = [4.0 12.0 -16.0; 12.0 37.0 -43.0; -33.0 -33.0 33.0]
@testset "Check for correct error handling" begin
	@test_throws Exception cholesky_decomposition(A2)
end
