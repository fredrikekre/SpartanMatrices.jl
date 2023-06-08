using SpartanMatrices
using Test

import SparseArrays

@testset "SpartanArrays" begin
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
    for (row, col) in ((1, 0), (0, 1), (m+1, 1), (1, n+1)), mat in (csc, csr)
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
