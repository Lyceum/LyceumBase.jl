function test(a = 10, b = 5, N = 10)
    x = [rand((1, 1, -1)) * (rand() * a)^(b * rand()) for _ = 1:N]
    s = KahanSum()
    for i = 1:N
        s += x[i]
    end
    sum_kbn(x) == sum(s)
end

let xs = [1.0, 10.0^100, 1.0, -10.0^100]
    s = KahanSum()
    s1 = KahanSum()
    s2 = KahanSum()

    s1 += xs[1]
    s1 += xs[2]
    s2 += xs[3]
    s2 += xs[4]
    for el in xs
        s += el
    end

    @test sum_kbn(xs) == sum(s1 + s2) == sum(s)
end

let pass = true
    for a = 1:2:10, b = 1:2:10, _ = 1:100
        pass &= test(a, b, 10000)
    end
    @test pass
end
