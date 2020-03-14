

abstract type TypedBool end

struct True <: TypedBool end
struct False <: TypedBool end

const Flags = Union{True, False, Val{true}, Val{false}, Bool}

TypedBool(flag::TypedBool) = flag
TypedBool(::Val{true}) = True()
TypedBool(::Val{false}) = False()
TypedBool(flag::Bool) = TypedBool(Val(flag))

untyped(::True) = true
untyped(::False) = false
untyped(flag::Flags) = untyped(TypedBool(flag))

not(::True) = False()
not(::False) = True()
not(flag::Flags) = not(TypedBool(flag))
