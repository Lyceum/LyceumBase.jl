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
        #alpha, beta = promote(alpha, beta, zero(T))
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
        #alpha, beta = promote(alpha, beta, zero(T))
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


mutable struct Converged{T}
    "absolute tolerance"
    tol::T
    lastval::T
    initialized::Bool
    Converged{T}(tol) where {T<:Number} = new{T}(T(tol), zero(T), false)
end
Converged(tol) = Converged{typeof(tol)}(tol)

function (x::Converged)(val::Real)
    ret = x.initialized && abs(val - x.lastval) < x.tol
    x.initialized = true
    x.lastval = val
    ret
end

reset!(x::Converged{T}) where {T} = (x.initialized = false; x.lastval = zero(T); x)




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
