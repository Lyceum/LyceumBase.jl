module Mod
const AbsArr{T,N} = AbstractArray{T,N}
const Dam{V,M,T,N} = AbsArr{T,N} where T <: AbsArr{V,M}
end