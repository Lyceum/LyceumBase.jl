abstract type Point3D{T} <: FieldVector{3, T} end

struct SPoint3D{T} <: Point3D{T}
    x::T
    y::T
    z::T
end

@inline function SPoint3D{T}(A::AbsMat, colidx::Integer) where {T}
    @boundscheck _checkbounds_point3d(A, colidx)
    @inbounds SPoint3D{T}(A[1, colidx], A[2, colidx], A[3, colidx])
end
@inline SPoint3D(A::AbsMat, colidx::Integer) = SPoint3D{eltype(A)}(A, colidx)


mutable struct MPoint3D{T} <: Point3D{T}
    x::T
    y::T
    z::T
end

@inline function MPoint3D{T}(A::AbsMat, colidx::Integer) where {T}
    @boundscheck _checkbounds_point3d(A, colidx)
    @inbounds MPoint3D{T}(A[1, colidx], A[2, colidx], A[3, colidx])
end
@inline MPoint3D(A::AbsMat, colidx::Integer) = MPoint3D{eltype(A)}(A, colidx)


@inline function _checkbounds_point3d(A::AbsMat, colidx::Integer)
    axes(A, 1) === Base.OneTo(3) || throw(ArgumentError("`A` must be a (3, N) matrix"))
    if !checkbounds(Bool, axes(A, 2), colidx)
        Base.throw_boundserror(A, (Base.OneTo(3), colidx))
    end
    nothing
end

