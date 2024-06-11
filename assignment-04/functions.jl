module Functions

export compute_features, square, cube, sqrt, exp

function compute_features(x, func::Vector{Function})
	return [f(x) for f in func]
end

function square(x)
	return x^2
end

function cube(x)
	return x^3
end

function sqrt(x)
	return sqrt(x)
end

function exp(x)
	return exp(x)
end

end # module Functions
