module TestSetfield

using LyceumBase: @set!
include("preamble.jl")

struct Imm
    x
    y
end

mutable struct Mut
    a
    b
end

@testset "nested mutable" begin
    let x = orig = Mut(Imm(Mut(1, 2), 3), 4)
        @set! x.a.x.a = 10
        @test x.a.x.a == 10
        @test x === orig
    end
    let x = Imm(Imm(Imm(1, 2), 3), 4)
        @test_throws ErrorException @set! x.x.x.x = 10
    end
end

@testset "arrays" begin
    let v = [1, 2, Mut(1, 2)], x = orig = Mut((x = v, y = 3), 4)
        @set! x.a.x[3].a = 10
        @test x.a.x[3].a == 10
        @test x === orig
    end
    let v = (1, 2, Imm(1, 2)), x = orig = Imm((x = v, y = 3), 4)
        @test_throws ErrorException @set! x.x.x[3].x = 10
    end
end

@testset "index lenses" begin
    @testset "ConstIndexLens" begin
        let v = [1, 2, 3]
            i = j = 1
            @set! v[$(i + j)] = 20
            @test v[2] == 20
        end
        let v = (1, 2, 3)
            i = j = 1
            @test_throws ErrorException @set! v[$(i + j)] = 20
        end
    end
    @testset "DynamicIndexLens" begin
        let v = [1, 2, 3]
            @set! v[end] = 30
            @test v[3] == 30
        end
        let v = (1, 2, 3)
            @test_throws ErrorException @set! v[end] = 30
        end
    end
end

end  # module
