# ruSOLVER — explicit numerical linear solvers

[Library index](README.md) · [中文](../../zh/libraries/rusolver.md)

**Experimental source.**

Cargo package `ruda-solver`, Rust crate `rusolver`, directory `ruSOLVER`. Defaults to dependency-free synchronous host FP64. GPU execution is opt-in, not a hidden fallback.

| Operation | API | Scope |
|---|---|---|
| Partial-row-pivot LU | `Lu::factor` | Nonempty square FP64 host matrices; P A = L U |
| Solve / transpose solve / inverse / signed log determinant | `Lu::solve/solve_transpose/inverse/slogdet` | Factor reuse; multiple RHS columns |
| Cholesky | `Cholesky::factor/solve/log_determinant` | Symmetric positive definite; no hidden jitter |
| Column-pivoted Householder QR / least squares | `Qr::factor/least_squares` | m>=n; solve requires full column rank; A[:,p]=Q R |
| Real symmetric eigensolver | `symmetric_eigen` | Scaled cyclic Jacobi; ascending values and column eigenvectors |
| Preconditioned CG | `conjugate_gradient` | Caller guarantees fixed SPD operator/preconditioner |
| ruSPARSE input adapter | `sparse::CsrF32Operator` | `sparse` feature; borrowed FP32 CSR -> FP64 host arithmetic |
| Native device batch solve | `tensor::cholesky_solve_batched` | `tensor`/`cuda`; small FP32 systems only |

```rust
use rusolver::{Matrix, Lu};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = Matrix::new(2, 2, vec![4.0, 1.0, 1.0, 3.0])?;
    let b = Matrix::new(2, 1, vec![6.0, 7.0])?;
    let factor = Lu::factor(a.view(), Default::default())?;
    let x = factor.solve(b.view())?;
    println!("{:?}", x.values());
    Ok(())
}
```

Matrices own finite nonempty row-major FP64 data. Immutable positive-stride views and explicit transposes are supported. FP32 widening is explicit. Prefer solving with reusable factors to forming an inverse. Existing tensor-level LU is not modified or automatically redirected.

Tolerances are `max(abs, rel*scale)`; LU uses max-absolute input scale, QR the largest initial column norm. Numerical singularity/rank decisions are not condition-number estimates. Cholesky/eigen check full symmetry; within tolerance the lower triangle defines the effective symmetric matrix. Rank-deficient least squares returns an error rather than a fake minimum-norm solution.

Scaled norms and compensated dot sums mitigate common rounding/overflow issues; unrepresentable intermediate products/results still return errors. CG can report numerical breakdown on extreme scales. It explicitly recomputes the true residual before success and periodically restarts after residual replacement. Check `report.converged()`; hitting the iteration limit is not success. Matvec/preconditioner callbacks must fully overwrite output and preserve a fixed operator throughout the solve.

## Reusable FP32/FP64 device factorizations

Feature `tensor` exposes real FP32/FP64 GPU operations separately from the synchronous host API. These native interfaces use `[batch,m,n]` matrices and matching `[batch,m,nrhs]` RHS tensors on the same device/queue; PyTorch's bridge expands/broadcasts its higher-rank batches before dispatch.

| Device API | Result/operation |
| --- | --- |
| `lu_factor_blocked(a, pivot, options)` | Rectangular LU; `LuFactorResult { packed_lu, pivots, info, submitted_kernels }`. Rust pivots are sequential zero-based row swaps with `P A=L U`. |
| `lu_solve_factor`, `lu_inverse_factor`, `lu_unpack_factor` | Reuse packed factors; solve accepts transpose. Unpack produces explicit P/L/U. |
| `cholesky_factor_blocked`, `cholesky_solve_factor`, `cholesky_inverse_factor` | Upper/lower selected-triangle factorization and factor reuse. |
| `triangular_solve`, `triangular_inverse` | Selected triangle, transpose and unit-diagonal device substitution. |
| `qr_factor_blocked`, `qr_form_q`, `qr_extract_r`, `qr_apply_q` | Packed Householder reflectors/tau; reduced/full Q, R extraction and Q application. This device QR is not the column-pivoted host `Qr`. |
| `least_squares_qr` | Full-rank tall least squares or wide minimum-norm solve using QR of the transpose; returns solution, residuals and info, without normal equations. |
| `determinant_from_lu`, `symmetric_eigen_device`, `svd_device` | Determinant/sign/log magnitude from factors; real Jacobi eigen/SVD device operations. |

Standalone LU/Cholesky factor results retain partial factors on failure and report `info`; their `check_status_sync()` explicitly checks device status. This does not change the older combined Cholesky solver's zeroed failed-output contract below. Native kernels enforce shape, layout, indexing and solver-option bounds and do not provide complex device factorizations.

The [PyTorch linalg backend](../native-pytorch-api.md#scientific-operations) registers standard solve/factorization, QR/eigen/SVD, inverse/determinant and least-squares APIs, with PyTorch's own autograd formulas. PyTorch pivots are converted to one-based int32. Its least-squares driver is only `None`/`gels`; SVD accepts `None` with native Jacobi, not named NVIDIA drivers. Full Q/full SVD bases are explicit, not thin substitutes.

Sources: [factorization](../../../ruSOLVER/src/tensor/factorization.rs), [factor reuse and least squares](../../../ruSOLVER/src/tensor/linear.rs), [Jacobi](../../../ruSOLVER/src/tensor/jacobi.rs), [Python registration](../../../ruda-torch/python/ruda_torch/linalg.py).

## Small batched Cholesky device contract

A `[batch,n,n]`, B `[batch,n,nrhs]`, n=1..32, nrhs=1..8. Same-device/queue, unquantized row-major contiguous FP32 only. One thread processes one matrix sequentially; matrices are independent across the batch. This is NOT a blocked, warp-cooperative, large-matrix performance implementation. Scratch is device memory, not exclusively registers.

One kernel per nonempty batch performs numeric validation, factorization and solves. Inputs are unchanged; factors/solutions/status are new allocations. Optional nonnegative `diagonal_shift` explicitly solves `(A+shift*I)X=B`. No automatic precision conversion or autograd.

Inspect `info` before using results: 0 success; positive one-based nonpositive pivot; -1 nonfinite input; -2 nonsymmetric; -3 nonfinite arithmetic. Failed results are initialized zeros, not valid solutions. `check_status_sync()` explicitly synchronizes/status-reads; no host numerical substitute. Launch success is not execution success.

```bash
cargo run --locked -p ruda-solver --example solver-demo
cargo run --locked -p ruda-solver --features sparse --example sparse-poisson
```

The [extension](science-extended.md) adds SVD, complex LU/QR/Cholesky, sparse LU, distributed CG, host autodiff and additional device solvers. It does not provide a general-complex eigensolver, distributed direct factorization or LAPACK/cuSOLVER ABI parity. See the [Chinese guide](../../zh/libraries/rusolver.md) for algorithm sources; methods are independently implemented rather than copied from those libraries.
