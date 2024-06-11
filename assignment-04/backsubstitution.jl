module BackSubstitution

export compute_y, compute_x

function compute_y(L::Matrix{T}, b::Vector{T}) where T <: AbstractFloat
	y = zeros(T, length(b))
    for i in 1:length(b)
        sum = 0.0
        for j in 1:i-1
            sum += L[i, j] * y[j]
        end
        y[i] = (b[i] - sum) / L[i, i]
    end
    return y
end

function compute_x(U::Matrix{T}, y::Vector{T}) where T <: AbstractFloat
	x = zeros(T, length(y))
    for i in length(y):-1:1
        sum = 0.0
        for j in i+1:length(y)
            sum += U[i, j] * x[j]
        end
        x[i] = (y[i] - sum) / U[i, i]
    end
    return x
end

end
