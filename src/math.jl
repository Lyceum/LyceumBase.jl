function scaleandcenter!(x::AbstractArray; center = 0, range = 1)
    minn, maxx = extrema(x)
    if maxx ≈ minn
        x .= center
    else
        @. x = ((x - minn) / (maxx - minn) - 1 // 2) * range + center
    end
    x
end


@inline function symmul!(
    C::Symmetric{T},
    transA::Transpose{<:Any,<:StridedVecOrMat{T}},
    B::StridedVecOrMat{T},
    alpha::Number = true,
    beta::Number = false,
) where {T<:LinearAlgebra.BlasFloat}
    A = transA.parent
    if A === B
        alpha = convert(T, alpha)
        beta = convert(T, beta)
        BLAS.syrk!(C.uplo, 'T', alpha, A, beta, C.data)
        return C
    else
        return LinearAlgebra.gemm_wrapper!(C, 'T', 'N', A, B, alpha, beta)
    end
end

@inline function symmul!(
    C::Symmetric{T},
    A::StridedVecOrMat{T},
    transB::Transpose{<:Any,<:StridedVecOrMat{T}},
    alpha::Number = true,
    beta::Number = false,
) where {T<:LinearAlgebra.BlasFloat}
    B = transB.parent
    if A === B
        alpha = convert(T, alpha)
        beta = convert(T, beta)
        BLAS.syrk!(C.uplo, 'N', alpha, B, beta, C.data)
        return C
    else
        return LinearAlgebra.gemm_wrapper!(C, 'N', 'T', A, B, alpha, beta)
    end
end


"""
    $(SIGNATURES)

Wrap the angle `theta`, specified in radians, to the interval [-pi, pi)
"""
@inline wraptopi(theta::Real) = mod1(theta + pi, 2pi) - pi


mutable struct Converged
    atol::Float64
    rtol::Maybe{Float64}
    nans::Bool
    lastval::Number
    initialized::Bool
    Converged(atol, rtol, nans) = new(atol, rtol, nans, 0, false)
end
Converged(; atol = 0, rtol = nothing, nans = false) = Converged(atol, rtol, nans)

function (x::Converged)(val::Number)
    if x.initialized
        rtol = x.rtol !== nothing ? x.rtol : Base.rtoldefault(x.lastval, val, x.atol)
        converged = isapprox(x.lastval, val, atol = x.atol, rtol = rtol, nans = x.nans)
    else
        x.initialized = true
        converged = false
    end
    x.lastval = val
    return converged
end

reset!(x::Converged) = (x.initialized = false; x.lastval = 0; x)


perturb(rng::AbstractRNG, A::AbstractArray) = perturb!(rng, copy(A))
perturb!(A::AbstractArray) = perturb!(Random.default_rng(), A)
function perturb!(rng::AbstractRNG, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += rand(rng)
    end
    A
end

perturbn(rng::AbstractRNG, A::AbstractArray) = perturb!(rng, copy(A))
perturbn!(A::AbstractArray) = perturb!(Random.default_rng(), A)
function perturbn!(rng::AbstractRNG, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += randn(rng)
    end
    A
end

perturb(rng::AbstractRNG, s::Sampleable, A::AbstractArray) = perturb!(rng, s, copy(A))
perturb!(s::Sampleable, A::AbstractArray) = perturb!(Random.default_rng(), s, A)
function perturb!(rng::AbstractRNG, s::Sampleable, A::AbstractArray)
    for i in eachindex(A)
        @inbounds A[i] += rand(rng, s)
    end
    A
end
