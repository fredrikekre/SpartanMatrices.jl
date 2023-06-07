module SpartanMatrices

export CSCMatrix, CSRMatrix
export cscmatrix, csrmatrix

import SparseArrays

# Note: consts instead of using Base: ... to please the linter
const var"@propagate_inbounds" = Base.var"@propagate_inbounds"
const Forward = Base.Order.Forward

struct CSCMatrix{Tv, Ti} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
end

struct CSRMatrix{Tv, Ti} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
end

const CSXMatrix{Tv, Ti} = Union{CSCMatrix{Tv, Ti}, CSRMatrix{Tv, Ti}}

function Base.size(csx::CSXMatrix)
    return (csx.m, csx.n)
end

function cscmatrix(args...)
    S = SparseArrays.sparse(args...)
    return CSCMatrix(S.m, S.n, S.colptr, S.rowval, S.nzval)
end

function csrmatrix(I, J, args...)
    S = SparseArrays.sparse(J, I, args...) # Note swap of I and J
    return CSRMatrix(S.n, S.m, S.colptr, S.rowval, S.nzval)
end

# This monstrosity is needed in order to bypass the inner constructor of
# SparseMatrixCSC (which has some expensive checks on the buffers).
let m      = Expr(:call, :getfield, :S, QuoteNode(:m)     ),
    n      = Expr(:call, :getfield, :S, QuoteNode(:n)     ),
    colptr = Expr(:call, :getfield, :S, QuoteNode(:colptr)),
    rowptr = Expr(:call, :getfield, :S, QuoteNode(:rowptr)),
    rowval = Expr(:call, :getfield, :S, QuoteNode(:rowval)),
    colval = Expr(:call, :getfield, :S, QuoteNode(:colval)),
    nzval  = Expr(:call, :getfield, :S, QuoteNode(:nzval) ),
    CSC    = Expr(:call, :getfield, :SparseArrays, QuoteNode(:SparseMatrixCSC))
    @eval begin
        # CSCMatrix -> SparseMatrixCSC
        function unsafe_cast(::Type{SparseArrays.SparseMatrixCSC}, S::CSCMatrix{Tv, Ti}) where {Tv, Ti}
            return $(Expr(:new, Expr(:curly, CSC, :Tv, :Ti), m, n, colptr, rowval, nzval))
        end
        # CSRMatrix -> SparseMatrixCSC (note that the result is transposed)
        function unsafe_cast(::Type{SparseArrays.SparseMatrixCSC}, S::CSRMatrix{Tv, Ti}) where {Tv, Ti}
            return $(Expr(:new, Expr(:curly, CSC, :Tv, :Ti), n, m, rowptr, colval, nzval))
        end
    end
end
# SparseMatrixCSC -> CSCMatrix
function unsafe_cast(::Type{CSCMatrix}, S::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    CSCMatrix{Tv, Ti}(S.m, S.n, S.colptr, S.rowval, S.nzval)
end
# SparseMatrixCSC -> CSRMatrix (note that the result is transposed)
function unsafe_cast(::Type{CSRMatrix}, S::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    CSRMatrix{Tv, Ti}(S.n, S.m, S.colptr, S.rowval, S.nzval)
end
# CSCMatrix -> CSRMatrix (note that the result is transposed)
function unsafe_cast(::Type{CSRMatrix}, S::CSCMatrix{Tv, Ti}) where {Tv, Ti}
    return CSRMatrix{Tv, Ti}(S.n, S.m, S.colptr, S.rowval, S.nzval)
end
# CSRMatrix -> CSCMatrix (note that the result is transposed)
function unsafe_cast(::Type{CSCMatrix}, S::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    return CSCMatrix{Tv, Ti}(S.n, S.m, S.rowptr, S.colval, S.nzval)
end

@propagate_inbounds function Base.getindex(A::CSCMatrix{Tv, Ti}, row::Int, col::Int) where {Tv, Ti}
    @boundscheck checkbounds(A, row, col)
    S = unsafe_cast(SparseArrays.SparseMatrixCSC, A)
    return @inbounds getindex(S, row, col)
end
@propagate_inbounds function Base.getindex(A::CSRMatrix{Tv, Ti}, row::Int, col::Int) where {Tv, Ti}
    @boundscheck checkbounds(A, row, col)
    S = unsafe_cast(SparseArrays.SparseMatrixCSC, A)
    return @inbounds getindex(S, col, row) # Note swap of col and row
end

end # module SpartanMatrices
