using SpartanMatrices
using Test

using SparseArrays: SparseArrays, sprand, findnz
using LinearAlgebra: LinearAlgebra, mul!

function aliased_sparsity_pattern(A::SpartanMatrices.CSXMatrix, B::SpartanMatrices.CSXMatrix)
    return SpartanMatrices.BaseType(A) === SpartanMatrices.BaseType(B) &&
        axes(A) == axes(B) &&
        SpartanMatrices.rowcolptr(A) === SpartanMatrices.rowcolptr(B) &&
        SpartanMatrices.rowcolval(A) === SpartanMatrices.rowcolval(B)
end

@testset "SpartanArrays basics" begin
    # Constructors
    I = [1, 1, 2, 3]
    J = [1, 2, 3, 4]
    V = rand(4)
    m = maximum(I)
    n = maximum(J)
    csc = cscmatrix(I, J, V)
    csr = csrmatrix(I, J, V)
    CSC = SparseArrays.sparse(I, J, V)
    @test csc == csr == CSC
    @test size(csc) == size(csr) == size(CSC)
    # Scalar getindex
    li = LinearIndices(CSC)
    for col in 1:n, row in 1:m
        @test csc[row, col] == csc[li[row, col]] ==
            csr[row, col] == csr[li[row, col]] ==
            CSC[row, col] == CSC[li[row, col]]
    end
    for (row, col) in ((1, 0), (0, 1), (m + 1, 1), (1, n + 1)), mat in (csc, csr)
        @test_throws BoundsError mat[row, col]
    end
    # Scalar setindex!
    row, col, val = 1, 1, 42
    for mat in (csc, csr)
        @test setindex!(mat, val, row, col) === mat
        @test mat[row, col] == val
    end
    row, col = 2, 1 # not stored
    for mat in (csc, csr)
        @test_throws SpartanMatrices.SparsityError setindex!(mat, val, row, col)
    end
end

@testset "SpartanArrays utilities" begin
    # SparsityError
    err = SpartanMatrices.SparsityError()
    @test sprint(showerror, err) == "SparsityError"
    m, n = (10, 20)
    err = SpartanMatrices.SparsityError(lazy"test ($m, $n)")
    @test sprint(showerror, err) == "SparsityError: test ($m, $n)"
end

@testset "A::CSXMatrix{$T} $op B::CSXMatrix{$T}" for T in (Float32, Float64), op in (+, -)
    for m in (10, 20), n in (10, 20)
        CSC = sprand(T, m, n, 0.1)
        CSR = transpose(CSC)
        csc = SpartanMatrices.unsafe_cast(CSCMatrix, CSC)
        csr = SpartanMatrices.unsafe_cast(CSRMatrix, CSC)
        for (csx, CSX) in ((csc, CSC), (csr, CSR))
            x = op(csx, csx)
            @test x == op(CSX, CSX)
            @test SpartanMatrices.rowcolptr(x) === SpartanMatrices.rowcolptr(csx)
            @test SpartanMatrices.rowcolval(x) === SpartanMatrices.rowcolval(csx)
        end
        CSC2 = sprand(T, m, n, 0.1)
        CSR2 = transpose(CSC2)
        csc2 = SpartanMatrices.unsafe_cast(CSCMatrix, CSC2)
        csr2 = SpartanMatrices.unsafe_cast(CSRMatrix, CSC2)
        @test_throws Exception op(csc, csc2)
        @test_throws Exception op(csr, csr2)
        @test_throws Exception op(csc, csr)
    end
end

@testset "A::CSXMatrix{$T} * b::Vector{$T}" for T in (Float32, Float64)
    n = 100
    CSC = sprand(T, n, n, 0.1)
    I, J, V = findnz(CSC)
    csc = cscmatrix(I, J, V, n, n)
    csr = csrmatrix(I, J, V, n, n)
    b = rand(T, n)
    # A * b
    @test csc * b ≈ csr * b ≈ CSC * b
    # mul!(c, A, b)
    @test mul!(similar(b), csc, b) ≈ mul!(similar(b), csr, b) ≈ mul!(similar(b), CSC, b)
    # mul!(c, A, b, α, β)
    α, β = 1.2, 3.4
    c = rand(n)
    @test mul!(copy(c), csc, b, α, β) ≈ mul!(copy(c), csr, b, α, β) ≈ mul!(copy(c), CSC, b, α, β)
end

@testset "CSXMatrix matrix scaling" begin
    T = Float64
    n = 10
    CSC = sprand(T, n, n, 0.1)
    I, J, V = findnz(CSC)
    csc = cscmatrix(I, J, V, n, n)
    csr = csrmatrix(I, J, V, n, n)
    b = rand(T)
    # CSC
    function csccheck(A, B)
        @test size(x) == size(X) && x.rowval == X.rowval && x.colptr == X.colptr && x.nzval ≈ X.nzval
    end
    let x = csc * b, X = CSC * b
        @test aliased_sparsity_pattern(x, csc)
        @test x ≈ SpartanMatrices.unsafe_cast(CSCMatrix, X)
    end
    let x = b * csc, X = b * CSC
        @test aliased_sparsity_pattern(x, csc)
        @test x ≈ SpartanMatrices.unsafe_cast(CSCMatrix, X)
    end
    let x = csc / b, X = CSC / b
        @test aliased_sparsity_pattern(x, csc)
        @test x ≈ SpartanMatrices.unsafe_cast(CSCMatrix, X)
    end
    let x = b \ csc, X = b \ CSC
        @test aliased_sparsity_pattern(x, csc)
        @test x ≈ SpartanMatrices.unsafe_cast(CSCMatrix, X)
    end
    # CSR
end

@testset "Broadcasting" begin
    T = Float64
    n = 10
    CSC = sprand(T, n, n, 0.1)
    I, J, V = findnz(CSC)
    csc = cscmatrix(I, J, V, n, n)
    csr = csrmatrix(I, J, V, n, n)
    b = rand(T)
    let x = csc .* b, y = csr .* b, X = CSC .* b
        @test aliased_sparsity_pattern(x, csc)
        @test aliased_sparsity_pattern(y, csr)
        @test x == y == X
    end
    let x = csc .* csc, y = csr .* csr, X = CSC .* CSC
        @test aliased_sparsity_pattern(x, csc)
        @test aliased_sparsity_pattern(y, csr)
        @test x == y == X
    end
    let x = csc .+ b .* csc, y = csr .+ b .* csr, X = CSC .+ b .* CSC
        @test aliased_sparsity_pattern(x, csc)
        @test aliased_sparsity_pattern(y, csr)
        @test x == y == X
    end
    # Index vectors of broadcasting aliases the first CSXMatrix
    csc′ = CSCMatrix(csc.m, csc.n, copy(csc.colptr), copy(csc.rowval), copy(csc.nzval))
    csr′ = CSRMatrix(csr.m, csr.n, copy(csr.rowptr), copy(csr.colval), copy(csr.nzval))
    @test !aliased_sparsity_pattern(csc, csc′)
    @test !aliased_sparsity_pattern(csr, csr′)
    let x = csc .* csc′, y = csr .* csr′, X = CSC .* CSC
        @test aliased_sparsity_pattern(x, csc)
        @test aliased_sparsity_pattern(y, csr)
        @test x == y == X
    end
    let x = csc′ .* csc, y = csr′ .* csr, X = CSC .* CSC
        @test aliased_sparsity_pattern(x, csc′)
        @test aliased_sparsity_pattern(y, csr′)
        @test x == y == X
    end
    # Error paths
    for csx in (csc, csr)
        @test_throws SpartanMatrices.SparsityError csx .+ 1.0
        @test_throws SpartanMatrices.SparsityError 1.0 .+ csx
        @test_throws SpartanMatrices.SparsityError cos.(csx)
    end
    @test_throws SpartanMatrices.SparsityError csc .+ csr
end

@testset "Linear algebra" begin
    T = Float64
    T = ComplexF64
    n = 10
    CSC = sprand(T, n, n, 0.1) + 5 * LinearAlgebra.I
    CSC = CSC + CSC'
    I, J, V = findnz(CSC)
    csc = cscmatrix(I, J, V, n, n)
    csr = csrmatrix(I, J, V, n, n)
    b = rand(T, n)
    let x = csc \ b, y = csr \ b, X = CSC \ b
        @test x ≈ X
        @test y ≈ X
    end
end
