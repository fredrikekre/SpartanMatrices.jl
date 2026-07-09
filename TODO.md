# TODO

Methods/features missing before SpartanArrays is useful in real-life sparse
linear algebra (assembling and solving sparse systems).

Key theme: since `CSXMatrix <: AbstractMatrix`, most unimplemented operations
already "work" via generic `AbstractArray` fallbacks that iterate all `m*n`
entries through `getindex`. So the gap is not correctness but that a *sparse*
type silently gives dense-cost behavior. Most fixes are cheap because they can
delegate through `unsafe_cast` to `SparseMatrixCSC`, like `mul!`/`lu` already do.

## Tier 1 — blocking for real use

- [ ] `transpose` / `adjoint` returning a zero-copy `CSXMatrix`
      (`transpose(::CSCMatrix)` -> `CSRMatrix` sharing buffers; `adjoint` needs
      conjugated `nzval` for complex). Unlocks `A'`, `A'b`, normal equations.
      Currently falls back to a lazy `Transpose` wrapper.
- [ ] Matrix-matrix multiply (`A * B`, 5-arg `mul!`). Produces a new pattern,
      which fits the strict philosophy (returns a fresh matrix).
- [ ] `mul!` for transposed/adjoint operands (`A'b`, `mul!(y, A', x)`) — needed
      by Krylov solvers (CG/GMRES/BiCGStab). Nearly free with zero-copy transpose.
- [ ] Standard sparse accessor API: `nnz`, `nonzeros`, `nzrange`,
      `rowvals`/`colvals`, `findnz`. One-liners over existing fields; without
      them there is no efficient way to iterate stored entries.
- [ ] Efficient `norm` (fold over `nzval`); today it is O(m*n) via `getindex`.
      Beware the generic-fallback trap for other reductions too.

## Tier 2 — important

- [ ] Public conversions: `SparseMatrixCSC(A)`, `sparse(A)`, `Matrix(A)`,
      `convert`. `unsafe_cast` is internal; users need a safe, documented bridge.
- [ ] `Diagonal` interactions: `Diagonal(d) * A`, `A * Diagonal(d)`, left/right
      diagonal scaling (FEM boundary conditions, Jacobi preconditioning). These
      can preserve the pattern, so they suit the strict type.
- [ ] Multiple-RHS solve `A \ B` (B a matrix) and `ldiv!`.
- [ ] Verify the CSR factorization path (`lu`/`cholesky` of the transposed
      `SparseMatrixCSC`) hits UMFPACK/CHOLMOD and does not densify the
      `Transpose` wrapper. (Non-symmetric `lu` tests added.)
- [ ] In-place scalar scaling: `lmul!`/`rmul!`/`ldiv!`/`rdiv!` over `nzval`.

## Tier 3 — nice to have

- [ ] `diag`, `tr`, `issymmetric`, `ishermitian`, `isposdef`.
- [ ] Structural constructors: build a pattern once and assemble into it
      repeatedly (`modifyindex!` is the `A[i,j] += v` primitive; add a
      documented entry point and maybe a pattern-only constructor).
- [ ] `zero(A)` / `fill!(A, 0)` (pattern-preserving), `hcat`/`vcat`/`blockdiag`/`kron`.
- [ ] Relax the `mul!` eltype constraint (the specialized path only fires when
      `c`, `A`, `b` share `T`; mixed precision silently takes the slow generic route).

## Design decision to make explicit

Define what "strict" means operationally. A clean rule: operations that *can*
preserve the sparsity pattern do so (and error if a value would need to
appear/disappear); operations that inherently produce a new pattern (`*`,
`transpose`, conversions) return a fresh matrix. This settles `+`/`.*`/scaling
(preserve) vs `*`/`\` (new) uniformly, and resolves the `getindex` `# TODO`
(return zero vs error for non-stored entries).
