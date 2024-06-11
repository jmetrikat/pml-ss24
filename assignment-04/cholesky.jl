module Cholesky

using LinearAlgebra
export cholesky_decomposition

function cholesky_decomposition(A::Matrix{T}) where T <: AbstractFloat
    eigenvalues = eigvals(A)
	if !issymmetric(A) || any(eigenvalues .< 0)
		throw(DomainError("Matrix is not symmetric positive semi-definite"))
	end

	L = zeros(T, size(A))

	for i in 1:size(A, 1)
		for j in 1:i
			if j == 1 && i == 1
				L[i, j] = sqrt(A[i, j])
			elseif j == 1 && i > 1
				L[i, j] = A[i, j] / L[1, 1]
			elseif i == j
				L[i, j] = sqrt(A[i, j] - sum(L[i, 1:j-1] .^ 2))
			else
				L[i, j] = (A[i, j] - sum(L[i, 1:j-1] .* L[j, 1:j-1])) / L[j, j]
			end
		end
	end

	return L
end

end # module Cholesky
