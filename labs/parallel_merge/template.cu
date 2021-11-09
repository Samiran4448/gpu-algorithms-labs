#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define TILE_SIZE 512

// Ceiling funciton for X / Y.
__host__ __device__ static inline int ceil_div(int x, int y) {
    return (x - 1) / y + 1;
}
/******************************************************************************
 GPU kernels
*******************************************************************************/

/*
 * Sequential merge implementation is given. You can use it in your kernels.
 */
__device__ void merge_sequential(float* A, int A_len, float* B, int B_len, float* C) {
    int i = 0, j = 0, k = 0;
    // if(blockIdx.x==0&&threadIdx.x==0){
    //   printf("From Merge sequential function\n");
    //   printf("A_len: %d\t", A_len);
    //   printf("B_len: %d\n", B_len);
    // }

    while ((i < A_len) && (j < B_len)) {
        C[k++] = A[i] <= B[j] ? A[i++] : B[j++];
    }

    if (i == A_len) {
        while (j < B_len) {
            C[k++] = B[j++];
        }
    } else {
        while (i < A_len) {
            C[k++] = A[i++];
        }
    }
}

//from wmh
__device__ int co_rank(const int k, float* A, int m, float* B, int n) {
  int i     = k < m ? k : m; // i = min(k,m)
  int j     = k - i;
  int i_low = 0 > (k - n) ? 0 : k - n; // i_low = max(0, k-n)
  int j_low = 0 > (k - m) ? 0 : k - m; // i_low = max(0,k-m)
  int delta;
  bool active = true;
  while (active) {
    if (i > 0 && j < n && A[i - 1] > B[j]) {
      delta = ((i - i_low + 1) >> 1); // ceil((i-i_low)/2)
      j_low = j;
      j     = j + delta;
      i     = i - delta;
    } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
      delta = ((j - j_low + 1) >> 1); // ceil((j-j_low)/2)
      i_low = i;
      i     = i + delta;
      j     = j - delta;
    } else
      active = false;
  }
  return i;
}

/*
 * Basic parallel merge kernel using co-rank function
 * A, A_len - input array A and its length
 * B, B_len - input array B and its length
 * C - output array holding the merged elements.
 *      Length of C is A_len + B_len (size pre-allocated for you)
 */
__global__ void gpu_merge_basic_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int elt = ceil_div(A_len + B_len, blockDim.x * gridDim.x);
    int k_curr = tid * elt;
    if (A_len + B_len < k_curr)
      k_curr = A_len + B_len;
    int k_next = k_curr + elt;
    if (A_len + B_len < k_next)
      k_next = A_len + B_len;
    
    //k_curr start for thread, k_next end for thread
    int i_curr = co_rank(k_curr, A, A_len, B, B_len);
    int i_next = co_rank(k_next, A, A_len, B, B_len);

    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

/*
 * Arguments are the same as gpu_merge_basic_kernel.
 * In this kernel, use shared memory to increase the reuse.
 */
__global__ void gpu_merge_tiled_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
    __shared__ float tileA[TILE_SIZE];
    __shared__ float tileB[TILE_SIZE];
    __shared__ float tileC[TILE_SIZE];

    __shared__ int blk_A_curr, blk_A_next;
    __shared__ int blk_B_curr, blk_B_next;

    int elb = ceil_div(A_len + B_len, gridDim.x);
    int blk_C_curr = blockIdx.x * elb;
    int blk_C_next = blk_C_curr + elb;

    blk_C_next     = (A_len + B_len < blk_C_next) ? A_len + B_len : blk_C_next;

    if(threadIdx.x==0){
        blk_A_curr = co_rank(blk_C_curr, A, A_len, B, B_len);
        blk_A_next = co_rank(blk_C_next, A, A_len, B, B_len);
        blk_B_curr = blk_C_curr - blk_A_curr;
        blk_B_next = blk_C_next - blk_A_next;
    }
    __syncthreads();
    int C_length = blk_C_next - blk_C_curr;
    int A_length = blk_A_next - blk_A_curr;
    int B_length = blk_B_next - blk_B_curr;

    int num_tiles = ceil_div(C_length, TILE_SIZE);
    int C_produced = 0, A_consumed = 0, B_consumed = 0;

    for (int counter = 0; counter < num_tiles; counter++) {
      for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
        if (i < A_length - A_consumed) {
          tileA[i] = A[blk_A_curr + i];
        }
        if (i < B_length - B_consumed) {
          tileB[i] = B[blk_B_curr + i];
        }
      }
      __syncthreads();
    //   if (threadIdx.x == 0&&blockIdx.x==0) {
    //     printf(" tileA[0]: %f\n tileB[0]: %f\n", tileA[0], tileB[0]);
    //   }
      int per_thread = TILE_SIZE / blockDim.x;
      int thr_C_curr = threadIdx.x * per_thread;
      int thr_C_next = thr_C_curr + per_thread;

      int C_remaining = C_length - C_produced;
      if (C_remaining < thr_C_curr) {
        thr_C_curr = C_remaining;
      }
      if (C_remaining < thr_C_next) {
        thr_C_next = C_remaining;
      }
      C_remaining    = (TILE_SIZE < C_remaining) ? TILE_SIZE : C_remaining;
      int A_in_tile  = (TILE_SIZE < A_length - A_consumed) ? TILE_SIZE : A_length - A_consumed;
      int B_in_tile  = (TILE_SIZE < B_length - B_consumed) ? TILE_SIZE : B_length - B_consumed;
      int thr_A_curr = co_rank(thr_C_curr, tileA, A_in_tile, tileB, B_in_tile);
      int thr_A_next = co_rank(thr_C_next, tileA, A_in_tile, tileB, B_in_tile);
      int thr_B_curr = thr_C_curr - thr_A_curr;
      int thr_B_next = thr_C_next - thr_A_next;
    //   if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("tileC[0]: %f\n", tileC[0]);
    //   }
      merge_sequential(&tileA[thr_A_curr], thr_A_next - thr_A_curr, &tileB[thr_B_curr], thr_B_next - thr_B_curr, &tileC[thr_C_curr]);
    //   if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("After merge_sequential tileC[0]: %f\n", tileC[0]);
    //   }
    //   __syncthreads();
      for (int i = threadIdx.x; i < C_remaining; i += blockDim.x) {
        C[blk_C_curr + C_produced + i] = tileC[i];
      }
      counter++;
      A_consumed += co_rank(C_remaining, tileA, A_in_tile, tileB, B_in_tile);
      C_produced += C_remaining;
      B_consumed = C_produced - A_consumed;
    }
}

/*
 * gpu_merge_circular_buffer_kernel is optional.
 * The implementation will be similar to tiled merge kernel.
 * You'll have to modify co-rank function and sequential_merge
 * to accommodate circular buffer.
 */
__global__ void gpu_merge_circular_buffer_kernel(float* A, int A_len, float* B, int B_len, float* C) {
    /* Your code here */
}

/******************************************************************************
 Functions
*******************************************************************************/

void gpu_basic_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_basic_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_tiled_merge(float* A, int A_len, float* B, int B_len, float* C) {
  const int numBlocks = (A_len + B_len - 1) / BLOCK_SIZE + 1;
  gpu_merge_tiled_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}

void gpu_circular_buffer_merge(float* A, int A_len, float* B, int B_len, float* C) {
    const int numBlocks = 128;
    gpu_merge_circular_buffer_kernel<<<numBlocks, BLOCK_SIZE>>>(A, A_len, B, B_len, C);
}
