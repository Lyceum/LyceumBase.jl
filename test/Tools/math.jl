@testset "mmtv!/mmtv" begin
    d1, d2 = 5, 10
    A = rand(d1, d2)
    x = rand(d1)
    y = rand(d1)
    alpha = rand()
    beta = rand()
    for alpha in (0, 0.5, 1), beta in (0, 0.5, 1)
        y_test = alpha * A * (A' * x) + beta * y
        @test isapprox(mmtv!(copy(y), A, x, alpha, beta), y_test)
    end
    for alpha in (0, 0.5, 1)
        y_test = alpha * A * (A' * x)
        @test isapprox(mmtv(A, x, alpha), y_test)
    end
end
