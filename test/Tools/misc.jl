let x = rand(100)
    x[1] = 0
    x[2] = 1
    scaleandcenter!(x, center = 100, range = 10)
    minn, maxx = extrema(x)
    @test minn ≈ 95
    @test maxx ≈ 105
end


let X = Symmetric(zeros(100, 100)),
    Y = rand(100, 10),
    X2 = Symmetric(zeros(100, 100)),
    Y2 = copy(Y)

    BLAS.syrk!('U', 'N', 1.0, Y, 0.0, X.data)
    symmul!(X2, Y2, transpose(Y2), 1.0, 0.0)
    @test X == X2
end

let X = Symmetric(zeros(10, 10)),
    Y = rand(100, 10),
    X2 = Symmetric(zeros(10, 10)),
    Y2 = copy(Y)

    BLAS.syrk!('U', 'T', 1.0, Y, 0.0, X.data)
    symmul!(X2, transpose(Y2), Y2, 1.0, 0.0)
    @test X == X2
end

@test wraptopi(pi) ≈ pi
@test wraptopi(-pi) ≈ pi
@test wraptopi(pi + 0.1) ≈ -pi + 0.1
@test wraptopi(-pi - 0.1) ≈ pi - 0.1

@testset "perturb/perturb!" begin
    @test let A = rand(100), B = copy(A), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        perturb!(r1, A) == B .+ rand(r2, length(B))
    end
    @test let A = rand(100), B = copy(A), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        perturbn!(r1, A) == B .+ randn(r2, length(B))
    end
    @test let A = rand(100), B = copy(A), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        d = Uniform(-0.1, 0.2)
        perturb!(r1, d, A) == B .+ rand(r2, d, length(B))
    end

    @test let A = rand(100), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        B = perturb(r2, A)
        perturb!(r1, A) == B
    end
    @test let A = rand(100), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        B = perturbn(r2, A)
        perturb!(r1, A) == B
    end
    @test let A = rand(100), r1 = MersenneTwister(1), r2 = MersenneTwister(1)
        d = Uniform(-0.1, 0.2)
        B = perturb(r2, d, A)
        perturb!(r1, d, A) == B
    end
end
