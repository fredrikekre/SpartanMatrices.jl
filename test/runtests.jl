using SpartanMatrices
using Test

using SparseArrays: SparseArrays, sprand, findnz
using LinearAlgebra: mul!

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

@testset "A::CSXMatrix{$T} + B::CSXMatrix{$T}" for T in (Float32, Float64)
    for m in (10, 20), n in (10, 20)
        CSC = sprand(T, m, n, 0.1)
        CSR = transpose(CSC)
        csc = SpartanMatrices.unsafe_cast(CSCMatrix, CSC)
        csr = SpartanMatrices.unsafe_cast(CSRMatrix, CSC)
        for (csx, CSX) in ((csc, CSC), (csr, CSR))
            x = csx + csx
            @test x == CSX + CSX
            @test SpartanMatrices.rowcolptr(x) === SpartanMatrices.rowcolptr(csx)
            @test SpartanMatrices.rowcolval(x) === SpartanMatrices.rowcolval(csx)
        end
        CSC2 = sprand(T, m, n, 0.1)
        CSR2 = transpose(CSC2)
        csc2 = SpartanMatrices.unsafe_cast(CSCMatrix, CSC2)
        csr2 = SpartanMatrices.unsafe_cast(CSRMatrix, CSC2)
        @test_throws Exception csc + csc2
        @test_throws Exception csr + csr2
        @test_throws Exception csc + csr
    end
end

@testset "A::CSXMatrix{$T} * b::Vector{$T}" for T in (Float32, Float64)
    n = 100
    CSC = sprand(T, n, n, 0.1)
    I, J, V = findnz(CSC)
    csc = cscmatrix(I, J, V)
    csr = csrmatrix(I, J, V)
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
