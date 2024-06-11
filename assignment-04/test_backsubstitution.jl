
include("back_substitution.jl")
include("cholesky.jl")
using .BackSubstitution: compute_y, compute_x
using .Cholesky: cholesky_decomposition
using Test

A = [8.0 5.0 2.0; 5.0 5.0 0.0; 2.0 0.0 8.0]
b = [1.0, 2.0, 3.0]

@testset "Back substitution test according to the exercise" begin
	L = cholesky_decomposition(A)
	y = compute_y(L, b)
	x = compute_x(Matrix(L'), y)
	@test isapprox(A * x, b)
end
