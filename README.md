# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


## 3.1 and 3.2 Diagnostics Output:
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (166)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (166)
-----------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                        |
        out: Storage,                                                                                |
        out_shape: Shape,                                                                            |
        out_strides: Strides,                                                                        |
        in_storage: Storage,                                                                         |
        in_shape: Shape,                                                                             |
        in_strides: Strides,                                                                         |
    ) -> None:                                                                                       |
        size = np.prod(out_shape)--------------------------------------------------------------------| #2
                                                                                                     |
        if out_strides == in_strides: ---------------------------------------------------------------| #0
            for i in prange(size):  -----------------------------------------------------------------| #1
                out[i] = fn(in_storage[i])                                                           |
        else:                                                                                        |
            out_shape_array = np.asarray(out_shape, dtype=np.int32)                                  |
            in_shape_array = np.asarray(in_shape, dtype=np.int32)                                    |
                                                                                                     |
            for i in prange(size): ------------------------------------------------------------------| #3
                local_out_index = np.empty_like(out_shape_array)                                     |
                to_index(i, out_shape_array, local_out_index)                                        |
                                                                                                     |
                local_in_index = np.empty_like(in_shape_array)                                       |
                broadcast_index(local_out_index, out_shape_array, in_shape_array, local_in_index)    |
                                                                                                     |
                out_pos = index_to_position(local_out_index, out_strides)                            |
                in_pos = index_to_position(local_in_index, in_strides)                               |
                                                                                                     |
                out[out_pos] = fn(in_storage[in_pos])                                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #0, #1, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (222)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (222)
------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                             |
        out: Storage,                                                                     |
        out_shape: Shape,                                                                 |
        out_strides: Strides,                                                             |
        a_storage: Storage,                                                               |
        a_shape: Shape,                                                                   |
        a_strides: Strides,                                                               |
        b_storage: Storage,                                                               |
        b_shape: Shape,                                                                   |
        b_strides: Strides,                                                               |
    ) -> None:                                                                            |
        size = np.prod(out_shape)---------------------------------------------------------| #9
                                                                                          |
        stride_aligned = (                                                                |
            out_strides == a_strides == b_strides-----------------------------------------| #5, 4
            and out_shape == a_shape == b_shape-------------------------------------------| #6, 7
        )                                                                                 |
                                                                                          |
        if stride_aligned:                                                                |
            for i in prange(size):--------------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                                   |
        else:                                                                             |
            out_shape_array = np.asarray(out_shape, dtype=np.int32)                       |
            a_shape_array = np.asarray(a_shape, dtype=np.int32)                           |
            b_shape_array = np.asarray(b_shape, dtype=np.int32)                           |
                                                                                          |
            for i in prange(size):--------------------------------------------------------| #10
                out_index = np.empty_like(out_shape_array)                                |
                a_index = np.empty_like(a_shape_array)                                    |
                b_index = np.empty_like(b_shape_array)                                    |
                                                                                          |
                to_index(i, out_shape_array, out_index)                                   |
                                                                                          |
                broadcast_index(out_index, out_shape_array, a_shape_array, a_index)       |
                broadcast_index(out_index, out_shape_array, b_shape_array, b_index)       |
                                                                                          |
                out_pos = index_to_position(out_index, out_strides)                       |
                a_pos = index_to_position(a_index, a_strides)                             |
                b_pos = index_to_position(b_index, b_strides)                             |
                                                                                          |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                     |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 7 parallel for-
loop(s) (originating from loops labelled: #9, #4, #5, #6, #7, #8, #10).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (287)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (287)
-------------------------------------------------------------------|loop #ID
    def _reduce(                                                   |
        out: Storage,                                              |
        out_shape: Shape,                                          |
        out_strides: Strides,                                      |
        a_storage: Storage,                                        |
        a_shape: Shape,                                            |
        a_strides: Strides,                                        |
        reduce_dim: int,                                           |
    ) -> None:                                                     |
        size = np.prod(out_shape)----------------------------------| #12
        out_shape_np = np.asarray(out_shape, dtype=np.int32)       |
        a_shape_np = np.asarray(a_shape, dtype=np.int32)           |
        reduce_size = a_shape[reduce_dim]                          |
                                                                   |
        for i in prange(size): ------------------------------------| #13
            out_index = np.empty_like(out_shape_np)                |
            a_index = np.empty_like(a_shape_np)                    |
            to_index(i, out_shape_np, out_index)                   |
            a_index[:] = out_index---------------------------------| #11
            a_index[reduce_dim] = 0                                |
                                                                   |
            out_pos = index_to_position(out_index, out_strides)    |
            a_pos = index_to_position(a_index, a_strides)          |
                                                                   |
            accumulator = a_storage[a_pos]                         |
                                                                   |
            for j in range(1, reduce_size):                        |
                a_index[reduce_dim] = j                            |
                a_pos = index_to_position(a_index, a_strides)      |
                accumulator = fn(accumulator, a_storage[a_pos])    |
                                                                   |
            out[out_pos] = accumulator                             |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #12, #13).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--13 is a parallel loop
   +--11 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--11 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--13 (parallel)
   +--11 (serial)



Parallel region 0 (loop #13) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#13).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (324)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (324)
------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                            |
    out: Storage,                                                       |
    out_shape: Shape,                                                   |
    out_strides: Strides,                                               |
    a_storage: Storage,                                                 |
    a_shape: Shape,                                                     |
    a_strides: Strides,                                                 |
    b_storage: Storage,                                                 |
    b_shape: Shape,                                                     |
    b_strides: Strides,                                                 |
) -> None:                                                              |
    """NUMBA tensor matrix multiply function.                           |
                                                                        |
    Should work for any tensor shapes that broadcast as long as         |
                                                                        |
    ```                                                                 |
    assert a_shape[-1] == b_shape[-2]                                   |
    ```                                                                 |
                                                                        |
    Optimizations:                                                      |
                                                                        |
    * Outer loop in parallel                                            |
    * No index buffers or function calls                                |
    * Inner loop should have no global writes, 1 multiply.              |
                                                                        |
                                                                        |
    Args:                                                               |
    ----                                                                |
        out (Storage): storage for `out` tensor                         |
        out_shape (Shape): shape for `out` tensor                       |
        out_strides (Strides): strides for `out` tensor                 |
        a_storage (Storage): storage for `a` tensor                     |
        a_shape (Shape): shape for `a` tensor                           |
        a_strides (Strides): strides for `a` tensor                     |
        b_storage (Storage): storage for `b` tensor                     |
        b_shape (Shape): shape for `b` tensor                           |
        b_strides (Strides): strides for `b` tensor                     |
                                                                        |
    Returns:                                                            |
    -------                                                             |
        None : Fills in `out`                                           |
                                                                        |
    """                                                                 |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0              |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0              |
                                                                        |
    out_batch_stride = out_strides[0] if len(out_shape) == 3 else 0     |
    batch_size = out_shape[0] if len(out_shape) == 3 else 1             |
    out_rows, out_cols = out_shape[-2], out_shape[-1]                   |
    inner_dim = a_shape[-1]                                             |
                                                                        |
    for batch in prange(batch_size): -----------------------------------| #14
        for i in range(out_rows):                                       |
            for j in range(out_cols):                                   |
                sum_value = 0.0                                         |
                for k in range(inner_dim):                              |
                    a_pos = (                                           |
                        batch * a_batch_stride                          |
                        + i * a_strides[-2]                             |
                        + k * a_strides[-1]                             |
                    )                                                   |
                    b_pos = (                                           |
                        batch * b_batch_stride                          |
                        + k * b_strides[-2]                             |
                        + j * b_strides[-1]                             |
                    )                                                   |
                    sum_value += a_storage[a_pos] * b_storage[b_pos]    |
                out_pos = (                                             |
                    batch * out_batch_stride                            |
                    + i * out_strides[-2]                               |
                    + j * out_strides[-1]                               |
                )                                                       |
                out[out_pos] = sum_value                                |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #14).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None




You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py