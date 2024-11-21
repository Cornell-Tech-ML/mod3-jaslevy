# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


## 3.1 and 3.2 Diagnostics Output:
<details>
<pre>
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (164)
------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                               |
        out: Storage,                                                                                       |
        out_shape: Shape,                                                                                   |
        out_strides: Strides,                                                                               |
        in_storage: Storage,                                                                                |
        in_shape: Shape,                                                                                    |
        in_strides: Strides,                                                                                |
    ) -> None:                                                                                              |
        size = np.prod(out_shape)---------------------------------------------------------------------------| #2
                                                                                                            |
        stride_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(out_shape, in_shape)    |
        if stride_aligned:                                                                                  |
            for i in prange(size):--------------------------------------------------------------------------| #3
                out[i] = fn(in_storage[i])                                                                  |
        else:                                                                                               |
            for i in prange(size):--------------------------------------------------------------------------| #4
                local_out_index = np.zeros(MAX_DIMS, dtype=np.int32)----------------------------------------| #0
                to_index(i, out_shape, local_out_index)                                                     |
                                                                                                            |
                local_in_index = np.zeros(MAX_DIMS, dtype=np.int32)-----------------------------------------| #1
                broadcast_index(                                                                            |
                    local_out_index, out_shape, in_shape, local_in_index                                    |
                )                                                                                           |
                                                                                                            |
                out_pos = index_to_position(local_out_index, out_strides)                                   |
                in_pos = index_to_position(local_in_index, in_strides)                                      |
                                                                                                            |
                out[out_pos] = fn(in_storage[in_pos])                                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #4, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--4 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--4 (parallel)
   +--0 (serial)
   +--1 (serial)



Parallel region 0 (loop #4) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#4).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (180) is hoisted out
 of the parallel loop labelled #4 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: local_out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (183) is hoisted out
 of the parallel loop labelled #4 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: local_in_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (219)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (219)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                                                                                  |
        out: Storage,                                                                                                                                                          |
        out_shape: Shape,                                                                                                                                                      |
        out_strides: Strides,                                                                                                                                                  |
        a_storage: Storage,                                                                                                                                                    |
        a_shape: Shape,                                                                                                                                                        |
        a_strides: Strides,                                                                                                                                                    |
        b_storage: Storage,                                                                                                                                                    |
        b_shape: Shape,                                                                                                                                                        |
        b_strides: Strides,                                                                                                                                                    |
    ) -> None:                                                                                                                                                                 |
                                                                                                                                                                               |
        size = np.prod(out_shape)----------------------------------------------------------------------------------------------------------------------------------------------| #9
        stride_aligned = (                                                                                                                                                     |
            np.array_equal(out_strides, a_strides) and np.array_equal(out_strides, b_strides) and np.array_equal(out_shape, a_shape) and np.array_equal(out_shape, b_shape)    |
        )                                                                                                                                                                      |
                                                                                                                                                                               |
        if stride_aligned:                                                                                                                                                     |
            for i in prange(size):---------------------------------------------------------------------------------------------------------------------------------------------| #8
                out[i] = fn(a_storage[i], b_storage[i])                                                                                                                        |
        else:                                                                                                                                                                  |
            for i in prange(size):---------------------------------------------------------------------------------------------------------------------------------------------| #10
                out_index = np.zeros(MAX_DIMS, dtype=np.int32)-----------------------------------------------------------------------------------------------------------------| #5
                a_i= np.zeros(MAX_DIMS, dtype=np.int32)------------------------------------------------------------------------------------------------------------------------| #6
                b_i= np.zeros(MAX_DIMS, dtype=np.int32)------------------------------------------------------------------------------------------------------------------------| #7
                                                                                                                                                                               |
                to_index(i, out_shape, out_index)                                                                                                                              |
                                                                                                                                                                               |
                broadcast_index(out_index, out_shape, a_shape, a_i)                                                                                                            |
                broadcast_index(out_index, out_shape, b_shape, b_i)                                                                                                            |
                                                                                                                                                                               |
                out_pos = index_to_position(out_index, out_strides)                                                                                                            |
                a_pos = index_to_position(a_i, a_strides)                                                                                                                      |
                b_pos = index_to_position(b_i, b_strides)                                                                                                                      |
                                                                                                                                                                               |
                out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])                                                                                                          |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--5 has the following loops fused into it:
   +--6 (fused)
   +--7 (fused)
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #9, #8, #10, #5).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--10 is a parallel loop
   +--5 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (parallel)
   +--6 (parallel)
   +--7 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--10 (parallel)
   +--5 (serial, fused with loop(s): 6, 7)



Parallel region 0 (loop #10) had 2 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#10).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (241) is hoisted out
 of the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (242) is hoisted out
 of the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_i= np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (243) is hoisted out
 of the parallel loop labelled #10 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_i= np.zeros(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (280)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (280)
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
        for i in prange(size):-------------------------------------| #13
            out_index = np.zeros_like(out_shape_np)                |
            a_i = np.zeros_like(a_shape_np)                        |
            to_index(i, out_shape_np, out_index)                   |
            a_i[:] = out_index-------------------------------------| #11
            a_i[reduce_dim] = 0                                    |
                                                                   |
            out_pos = index_to_position(out_index, out_strides)    |
            a_pos = index_to_position(a_i, a_strides)              |
                                                                   |
            accumulator = a_storage[a_pos]                         |
                                                                   |
            for j in range(1, reduce_size):                        |
                a_i[reduce_dim] = j                                |
                a_pos = index_to_position(a_i, a_strides)          |
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
/Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (316)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/jlevy/Desktop/MLE/mod3-jaslevy/minitorch/fast_ops.py (316)
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                |
    out: Storage,                                                                           |
    out_shape: Shape,                                                                       |
    out_strides: Strides,                                                                   |
    a_storage: Storage,                                                                     |
    a_shape: Shape,                                                                         |
    a_strides: Strides,                                                                     |
    b_storage: Storage,                                                                     |
    b_shape: Shape,                                                                         |
    b_strides: Strides,                                                                     |
) -> None:                                                                                  |
    """NUMBA tensor matrix multiply function.                                               |
                                                                                            |
    Should work for any tensor shapes that broadcast as long as                             |
                                                                                            |
    ```                                                                                     |
    assert a_shape[-1] == b_shape[-2]                                                       |
    ```                                                                                     |
                                                                                            |
    Optimizations:                                                                          |
                                                                                            |
    * Outer loop in parallel                                                                |
    * No index buffers or function calls                                                    |
    * Inner loop should have no global writes, 1 multiply.                                  |
                                                                                            |
                                                                                            |
    Args:                                                                                   |
    ----                                                                                    |
        out (Storage): storage for `out` tensor                                             |
        out_shape (Shape): shape for `out` tensor                                           |
        out_strides (Strides): strides for `out` tensor                                     |
        a_storage (Storage): storage for `a` tensor                                         |
        a_shape (Shape): shape for `a` tensor                                               |
        a_strides (Strides): strides for `a` tensor                                         |
        b_storage (Storage): storage for `b` tensor                                         |
        b_shape (Shape): shape for `b` tensor                                               |
        b_strides (Strides): strides for `b` tensor                                         |
                                                                                            |
    Returns:                                                                                |
    -------                                                                                 |
        None : Fills in `out`                                                               |
                                                                                            |
    """                                                                                     |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  |
                                                                                            |
    out_batch_stride = out_strides[0] if len(out_shape) == 3 else 0                         |
    batch_size = out_shape[0] if len(out_shape) == 3 else 1                                 |
    out_rows, out_cols = out_shape[-2], out_shape[-1]                                       |
    inner_dim = a_shape[-1]                                                                 |
                                                                                            |
    for batch in prange(batch_size):--------------------------------------------------------| #14
        for i in range(out_rows):                                                           |
            for j in range(out_cols):                                                       |
                sum_value = 0.0                                                             |
                for k in range(inner_dim):                                                  |
                    a_pos = (                                                               |
                        batch * a_batch_stride + i * a_strides[-2] + k * a_strides[-1]      |
                    )                                                                       |
                    b_pos = (                                                               |
                        batch * b_batch_stride + k * b_strides[-2] + j * b_strides[-1]      |
                    )                                                                       |
                    sum_value += a_storage[a_pos] * b_storage[b_pos]                        |
                out_pos = (                                                                 |
                    batch * out_batch_stride + i * out_strides[-2] + j * out_strides[-1]    |
                )                                                                           |
                out[out_pos] = sum_value                                                    |
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
</pre>
</details>

## 3.4 fast_ops vs cuda_ops Runtime Analysis
![image](https://github.com/user-attachments/assets/cbde856f-4a18-45b5-91f4-3b87f265b2d4)

The results show that the cuda_ops matmul implementation provides faster runtimes for larger matrices than the fast_ops matmul implementation

## 3.5 - CPU vs GPU training
### Small Models (hidden size 100, learning rate 0.05)

#### Simple
##### GPU
<img width="388" alt="Simple_results_mod4_gpu_small" src="https://github.com/user-attachments/assets/57fbf375-ac07-4097-aed9-d17ab6ff534f">

##### CPU
<img width="369" alt="Simple_results_mod4_cpu_small" src="https://github.com/user-attachments/assets/2ba08ef1-da4c-43bb-b3f8-fc34c3a85166">

#### XOR
##### GPU
<img width="366" alt="Xor_results_mod4_gpu_small" src="https://github.com/user-attachments/assets/c93cd10b-5f8e-4e4d-9ec1-441735fcbc91">

##### CPU
<img width="371" alt="Xor_results_mod4_cpu_small" src="https://github.com/user-attachments/assets/4c58eac7-1ec6-4b36-bcde-a63ad9f23633">

#### Split
##### GPU
<img width="364" alt="Split_results_mod4_gpu_small" src="https://github.com/user-attachments/assets/9d8e2c77-b29b-42f2-b7f2-5c737f3fb347">

##### CPU
<img width="378" alt="Split_results_mod4_cpu_small" src="https://github.com/user-attachments/assets/b16bce5b-f4df-4ba8-aadf-f08d9bae89b4">

### Large Model on Simple Dataset (hidden size 200, learning rate 0.05)
##### GPU
<img width="388" alt="Large_simple_results_mod4_gpu" src="https://github.com/user-attachments/assets/866337a0-9608-45cc-91ce-8ef5dd132dd7">

##### CPU
<img width="372" alt="Large_simple_results_mod4_cpu" src="https://github.com/user-attachments/assets/72f3c84c-8a9a-4ce0-838c-81a2e6a9bc41">


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
