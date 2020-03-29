using LyceumDevTools
using Test: Test, AbstractTestSet, DefaultTestSet
using Test: Result, Pass, Fail, Error, Broken
using Distributed: Distributed

const PROGRESS_SYMBOL = '●'
const LINE_WIDTH = 80
const LINE_SEP = '='

export @includetests ProgressTestSet


mutable struct ProgressTestSet{T<:AbstractTestSet} <: AbstractTestSet
    wrapped::T
    ProgressTestSet{T}(description) where {T} = new(T(description))
end

ProgressTestSet(desc; wrapped=DefaultTestSet) = ProgressTestSet{wrapped}(desc)

function Test.record(ts::ProgressTestSet, res::Union{Fail, Error})
    Distributed.myid() == 1 && printsep(color = Base.error_color())
    Test.record(ts.wrapped, res)
    Distributed.myid() == 1 && printsep(color = Base.error_color())
    res, isa(res, Error) || backtrace()
end

function Test.record(ts::ProgressTestSet, res::Pass)
    Test.record(ts.wrapped, res)
    printprog(color = :green)
    ts
end

function Test.record(ts::ProgressTestSet, res::Broken)
    Test.record(ts.wrapped, res)
    printprog(color = Base.warn_color())
    ts
end

Test.record(ts::ProgressTestSet, child::AbstractTestSet) = Test.record(ts.wrapped, child)

function Test.finish(ts::ProgressTestSet)
    Test.get_testset_depth() == 0 && print("\n\n")
    Test.finish(ts.wrapped)
end

printsep(;kwargs...) = printstyled('\n', repeat(LINE_SEP, LINE_WIDTH), '\n'; kwargs...)
printprog(;kwargs...) = printstyled(PROGRESS_SYMBOL; kwargs...)


ismatch(reg::Regex, s::AbstractString) = match(reg, s) !== nothing
ismatch(reg::Regex) = Base.Fix1(ismatch, reg)

macro includetests(testset, testname)
    ex = quote
        if $isempty(ARGS)
            local testfiles = $filter(f -> $ismatch(r"^test_.*\.jl$", $basename(f)), $flattendir($pwd(), dirs=false, join=false))
        else
            local testfiles = map(f -> endswith(f, ".jl") ? f : "$f.jl", ARGS)
        end
        @includetests $testset $testname testfiles
    end
    esc(ex)
end

macro includetests(testset, testname, testfiles)
    ex = quote
        local testname = $testname
        $Test.@testset $testset "$testname" begin
            for file in $testfiles
                $Test.@testset "$file" begin $Base.include($__module__, file) end
            end
        end
    end
    esc(ex)
end