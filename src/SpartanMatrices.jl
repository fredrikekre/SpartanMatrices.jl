module SpartanMatrices

export CSCMatrix, CSRMatrix
export cscmatrix, csrmatrix

using Base: Broadcast
using LinearAlgebra: LinearAlgebra, cholesky, lu, mul!, transpose
using SparseArrays: SparseArrays, SparseMatrixCSC

# Silence of the Langs(erver)
const var"@propagate_inbounds" = Base.var"@propagate_inbounds"
const Forward = Base.Order.Forward


###########
# Structs #
###########

struct CSCMatrix{Tv <: Number, Ti <: Integer} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    nzval::Vector{Tv}
end

struct CSRMatrix{Tv <: Number, Ti <: Integer} <: AbstractMatrix{Tv}
    m::Int
    n::Int
    rowptr::Vector{Ti}
    colval::Vector{Ti}
    nzval::Vector{Tv}
end

const CSXMatrix{Tv, Ti} = Union{CSCMatrix{Tv, Ti}, CSRMatrix{Tv, Ti}}


################
# Type casting #
################

# This monstrosity is needed in order to bypass the inner constructor of
# SparseMatrixCSC (which has some expensive checks on the buffers).
let m = Expr(:call, :getfield, :S, QuoteNode(:m)),
        n = Expr(:call, :getfield, :S, QuoteNode(:n)),
        colptr = Expr(:call, :getfield, :S, QuoteNode(:colptr)),
        rowptr = Expr(:call, :getfield, :S, QuoteNode(:rowptr)),
        rowval = Expr(:call, :getfield, :S, QuoteNode(:rowval)),
        colval = Expr(:call, :getfield, :S, QuoteNode(:colval)),
        nzval = Expr(:call, :getfield, :S, QuoteNode(:nzval))
    @eval begin
        # CSCMatrix -> SparseMatrixCSC
        @inline function unsafe_cast(::Type{SparseMatrixCSC}, S::CSCMatrix{Tv, Ti}) where {Tv, Ti}
            return $(Expr(:new, Expr(:curly, SparseMatrixCSC, :Tv, :Ti), m, n, colptr, rowval, nzval))
        end
        # CSRMatrix -> SparseMatrixCSC (note that the result is transposed)
        @inline function unsafe_cast(::Type{SparseMatrixCSC}, S::CSRMatrix{Tv, Ti}) where {Tv, Ti}
            return $(Expr(:new, Expr(:curly, SparseMatrixCSC, :Tv, :Ti), n, m, rowptr, colval, nzval))
        end
    end
end
# SparseMatrixCSC -> CSCMatrix
@inline function unsafe_cast(::Type{CSCMatrix}, S::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return CSCMatrix{Tv, Ti}(S.m, S.n, S.colptr, S.rowval, S.nzval)
end
# SparseMatrixCSC -> CSRMatrix (note that the result is transposed)
@inline function unsafe_cast(::Type{CSRMatrix}, S::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    return CSRMatrix{Tv, Ti}(S.n, S.m, S.colptr, S.rowval, S.nzval)
end
# CSCMatrix -> CSRMatrix (note that the result is transposed)
@inline function unsafe_cast(::Type{CSRMatrix}, S::CSCMatrix{Tv, Ti}) where {Tv, Ti}
    return CSRMatrix{Tv, Ti}(S.n, S.m, S.colptr, S.rowval, S.nzval)
end
# CSRMatrix -> CSCMatrix (note that the result is transposed)
@inline function unsafe_cast(::Type{CSCMatrix}, S::CSRMatrix{Tv, Ti}) where {Tv, Ti}
    return CSCMatrix{Tv, Ti}(S.n, S.m, S.rowptr, S.colval, S.nzval)
end


##################
# Misc utilities #
##################

struct SparsityError <: Exception
    msg::Union{Nothing, LazyString}
end
SparsityError() = SparsityError(nothing)

function Base.showerror(io::IO, e::SparsityError)
    print(io, "SparsityError")
    if e.msg !== nothing
        print(io, ": ", e.msg)
    end
    return
end

# TODO: unsafe_findtoken for when we know the entry is stored?
@propagate_inbounds function findtoken(A::CSCMatrix, row::Int, col::Int)
    @boundscheck checkbounds(A, row, col)
    kl = Int(A.colptr[col])
    ku = Int(A.colptr[col + 1] - 1)
    if ku < kl
        return nothing
    end
    k = searchsortedfirst(A.rowval, row, kl, ku, Forward)
    if k <= ku && A.rowval[k] == row
        return k
    else
        return nothing
    end
end
@propagate_inbounds function findtoken(A::CSRMatrix, row::Int, col::Int)
    return findtoken(unsafe_cast(CSCMatrix, A), col, row)
end


################
# Constructors #
################

function cscmatrix(I::AbstractVector, J::AbstractVector, args...)
    S = SparseArrays.sparse(I, J, args...)
    return unsafe_cast(CSCMatrix, S)
end

function csrmatrix(I::AbstractVector, J::AbstractVector, args...)
    S = SparseArrays.sparse(J, I, args...) # Note the swap of I and J
    return unsafe_cast(CSRMatrix, S)
end


###################
# Pretty printing #
###################

function Base.show(io::IO, ::MIME"text/plain", A::CSXMatrix)
    summary(io, A)
    nnz = length(A.nzval)
    println(io, " with $nnz stored entries:")
    if A isa CSCMatrix
        Base.print_matrix(io, unsafe_cast(SparseMatrixCSC, A))
    else
        Base.print_matrix(io, transpose(unsafe_cast(SparseMatrixCSC, A)))
    end
    return
end


################
# Base methods #
################

# This is reached from `copy` which calls `copymutable` (i.e. `similar`) and then `copyto!`.
function Base.copyto!(dst::CSXMatrix, src::CSXMatrix)
    require_same_sparsity_pattern(dst, src)
    copyto!(dst.nzval, src.nzval)
    return dst
end

###########################
# AbstractArray interface #
###########################

function Base.size(csx::CSXMatrix)
    return (csx.m, csx.n)
end

function Base.similar(A::CSXMatrix, ::Type{T}, dims::NTuple{N, Int}) where {T, N}
    if dims != size(A)
        throw(ArgumentError("size mismatch"))
    end
    return similar_with_vals(A, similar(A.nzval, T))
end

# Scalar getindex
# TODO: Error instead of returning zero for non-stored entries?
@propagate_inbounds function Base.getindex(A::CSXMatrix{Tv, Ti}, row::Int, col::Int) where {Tv, Ti}
    @boundscheck checkbounds(A, row, col)
    k = @inbounds findtoken(A, row, col)
    if k === nothing
        return zero(Tv)
    end
    return A.nzval[k]
end

# Scalar setindex!: only allowed if entry is already stored
# TODO: Allow if v === zero(T) even if entry not stored?
@propagate_inbounds function Base.setindex!(A::CSXMatrix{T}, v, row, col) where {T}
    return setindex!(A, T(v), Int(row), Int(col))
end
@propagate_inbounds function Base.setindex!(A::CSXMatrix{T}, v::T, row::Int, col::Int) where {T}
    @boundscheck checkbounds(A, row, col)
    k = @inbounds findtoken(A, row, col)
    if k === nothing
        throw(SparsityError(lazy"row = $row and col = $col is not stored"))
    end
    A.nzval[k] = v
    return A
end

# Note: not part of the AbstractArray interface
function modifyindex!(A::CSXMatrix, f::F, v, row, col) where {F}
    k = findtoken(A, row, col)
    if k === nothing
        throw(SparsityError(lazy"row = $row and col = $col is not stored"))
    end
    A.nzval[k] = f(A.nzval[k], v)
    return A
end

#############
# Broadcast #
#############

getindex_at_token(x::Number, _) = x
getindex_at_token(x::CSXMatrix, token) = x.nzval[token]

zero_if_csx(x::Number) = x
zero_if_csx(::CSXMatrix{Tv}) where {Tv} = zero(Tv)

# Broadcast style. Not differentiating between CSCMatrix and CSRMatrix here because we have
# to check it when instantiating the broadcasted object anyway.
struct SpartanStyle <: Broadcast.BroadcastStyle end
Broadcast.BroadcastStyle(::Type{<:CSXMatrix}) = SpartanStyle()

# Broadcasting is only allowed between CSXMatrices and numbers so only define promotion
# between these styles
Broadcast.BroadcastStyle(::SpartanStyle, ::Broadcast.DefaultArrayStyle{0}) = SpartanStyle()
Broadcast.BroadcastStyle(::SpartanStyle, ::SpartanStyle) = SpartanStyle()

# Return the output matrix
function Base.similar(bc::Broadcast.Broadcasted{SpartanStyle}, ::Type{T}) where {T}
    # TODO: This is done again in copyto!
    bc = Broadcast.flatten(bc)
    idx = findfirst(x -> x isa CSXMatrix, bc.args)
    if idx === nothing
        throw(ErrorException("unreachable"))
    end
    A = bc.args[idx]
    return similar(A, T, axes(bc))
end

# Materialization of the broadcasted object
function Base.copyto!(csx::CSXMatrix{Tv, Ti}, bc::Broadcast.Broadcasted{SpartanStyle}) where {Tv, Ti}
    bc = Broadcast.flatten(bc)
    # Verify that the function is preserving zeros
    z = bc.f(map(arg -> zero_if_csx(arg), bc.args)...)
    if z != zero(Tv)
        throw(SparsityError("broadcast kernel is not zero preserving"))
    end
    # Verify that all CSXMatrices have the same base type (CSCMatrix or CSRMatrix) and
    # sparsity pattern
    for arg in bc.args
        if arg isa CSXMatrix
            require_same_sparsity_pattern(csx, arg)
        end
    end
    # Loop over the stored entries and apply the kernel
    for token in eachindex(csx.nzval)
        csx.nzval[token] = bc.f(map(arg -> getindex_at_token(arg, token), bc.args)...)
    end
    return csx
end


##################
# Linear algebra #
##################

BaseType(A::CSXMatrix) = BaseType(typeof(A))
BaseType(::Type{<:CSCMatrix}) = CSCMatrix
BaseType(::Type{<:CSRMatrix}) = CSRMatrix
rowcolptr(A::CSCMatrix) = A.colptr
rowcolptr(A::CSRMatrix) = A.rowptr
rowcolval(A::CSCMatrix) = A.rowval
rowcolval(A::CSRMatrix) = A.colval

function same_sparsity_pattern(A::CSXMatrix, B::CSXMatrix)
    BaseType(A) === BaseType(B) || return false
    axes(A) == axes(B) || return false
    rcptrA = rowcolptr(A)
    rcvalA = rowcolval(A)
    rcptrB = rowcolptr(B)
    rcvalB = rowcolval(B)
    if rcptrA === rcptrB && rcvalA === rcvalB
        return true
    elseif rcptrA === rcptrB
        return rcvalA == rcvalB
    elseif rcvalA === rcvalB
        return rcptrA == rcptrB
    else
        return rcptrA == rcptrB && rcvalA == rcvalB
    end
end

function require_same_sparsity_pattern(A::CSXMatrix, B::CSXMatrix)
    # This check is also done in same_sparsity_pattern but we want to give a better error
    # message here
    if BaseType(A) !== BaseType(B)
        throw(SparsityError("operation doesn't support mixing CSCMatrix and CSRMatrix"))
    end
    if !same_sparsity_pattern(A, B)
        throw(SparsityError("the sparsity pattern of the two arrays are not the same"))
    end
    return
end

# Construct a new CSXMatrix A′ with the same sparsity pattern as A but new value vector.
# A.colptr/A.rowptr and A.rowval/A.colval are reused for A′.
function similar_with_vals(A::CSCMatrix{<:Any, Ti}, values::AbstractVector{Tv}) where {Tv, Ti}
    return CSCMatrix{Tv, Ti}(A.m, A.n, A.colptr, A.rowval, values)
end
function similar_with_vals(A::CSRMatrix{<:Any, Ti}, values::AbstractVector{Tv}) where {Tv, Ti}
    return CSRMatrix{Tv, Ti}(A.m, A.n, A.rowptr, A.colval, values)
end

# Matrix scaling
Base.:*(A::CSXMatrix, b::Number) = broadcast(*, A, b)
Base.:*(b::Number, A::CSXMatrix) = broadcast(*, b, A)
Base.:/(A::CSXMatrix, b::Number) = broadcast(/, A, b)
Base.:\(b::Number, A::CSXMatrix) = broadcast(\, b, A)

# Matrix addition/subtraction
Base.:+(A::CSXMatrix, B::CSXMatrix) = broadcast(+, A, B)
Base.:-(A::CSXMatrix, B::CSXMatrix) = broadcast(-, A, B)

# Matrix-vector multiplication
# TODO: Is this the correct place to hook into?
function LinearAlgebra.mul!(c::AbstractVector{T}, A::CSCMatrix{T}, b::AbstractVector{T}, α::Number, β::Number) where {T}
    return mul!(c, unsafe_cast(SparseMatrixCSC, A), b, α, β)
end
function LinearAlgebra.mul!(c::AbstractVector{T}, A::CSRMatrix{T}, b::AbstractVector{T}, α::Number, β::Number) where {T}
    return mul!(c, transpose(unsafe_cast(SparseMatrixCSC, A)), b, α, β)
end

# Factorizations
Base.:\(A::CSXMatrix, b::AbstractVector) = lu(A) \ b
LinearAlgebra.lu(A::CSCMatrix) = lu(unsafe_cast(SparseMatrixCSC, A))
LinearAlgebra.lu(A::CSRMatrix) = lu(transpose(unsafe_cast(SparseMatrixCSC, A)))
LinearAlgebra.cholesky(A::CSCMatrix) = cholesky(unsafe_cast(SparseMatrixCSC, A))
LinearAlgebra.cholesky(A::CSRMatrix) = cholesky(transpose(unsafe_cast(SparseMatrixCSC, A)))

end # module SpartanMatrices
