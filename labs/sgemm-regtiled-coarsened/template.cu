#include <cstdio>
#include <cstdlib>

#include "template.hu"

#define TILE_SZ_A 128
#define TILE_SZ_B 16
#define TILE_SZ_RATIO (TILE_SZ_A / TILE_SZ_B)

#define U TILE_SZ_B
#define T TILE_SZ_A
#define S TILE_SZ_RATIO

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float *C) {

/********************************************************************
 *
 * Compute C = A x B
 *   where A is a (m x k) matrix
 *   where B is a (k x n) matrix
 *   where C is a (m x n) matrix
 *
 * Use register and shared memory tiling and thread coarsening
 *
 * NOTE: A and C are column major, B is row major
 *
 ********************************************************************/

// Macros for accessing flattened matrices
#define A(row, col) A[(row) + (col) *m]
#define B(row, col) B[(row) *n + (col)]
#define C(row, col) C[(row) + (col) *m]

  // INSERT KERNEL CODE HERE

  // SSL Hint (9/6/21): try using just one register for the tile of A
  // rather than several--in other words, load one value (per thread)
  // from A and compute using that value rather than loading all values
  // before doing the computation.  This approach seems to be slightly
  // faster than the alternative.
  int ty          = threadIdx.y;
  int bx          = blockIdx.x;
  int by          = blockIdx.y;
  int tile_row    = ty / U;
  int tile_column = ty % U;

  // if (bx == 0 && by == 0)
  //   printf("I am thread %d, kernel was launched\n", threadIdx.y);
  __shared__ float B_shared[S][U];
  // float A_reg[S];
  float A_reg = 0;
  float C_reg[U];

  for (int i = 0; i < U; i++) {
    C_reg[i] = 0;
  }

  //__syncthreads();
  // load into memory (load full tile of B_shared to avoid excessive __syncthreads())
  for (int t = 0; t < (k + S - 1) / S; t++) {
    if ((t * S + tile_row) < k && (bx * U + tile_column) < n)
      B_shared[tile_row][tile_column] = B(t * S + tile_row, bx * U + tile_column);
    else
      B_shared[tile_row][tile_column] = 0;
    __syncthreads();
    for (int i = 0; i < S; i++) {
      if ((by * blockDim.y + ty) < m && (t * S + i) < k)
        A_reg = A(by * blockDim.y + ty, t * S + i);
      else
        A_reg = 0;
      // }

      for (int j = 0; j < U; j++) {
        // for (int i = 0; i < S; i++) {
        C_reg[j] += A_reg * B_shared[i][j];
        // }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < U; i++) {
    if ((by * blockDim.y + ty) < m && (bx * U + i) < n)
      C(by * blockDim.y + ty, bx * U + i) = C_reg[i];
  }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta,
                float *C, int ldc) {
  if ((transa != 'N') && (transa != 'n')) {
    printf("unsupported value of 'transa'\n");
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    printf("unsupported value of 'transb'\n");
    return;
  }

  if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
    printf("unsupported value of alpha\n");
    return;
  }

  if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
    printf("unsupported value of beta\n");
    return;
  }

  // Initialize thread block and kernel grid dimensions ---------------------

  // Your code need only consider the m, n, k, A, B, and C parameters of
  // the function, which provide the matrix sizes (m, n, k) and data
  // (A, B, C).

  // INSERT CODE HERE
  dim3 blockDim = dim3(1, T, 1);
  dim3 gridDim  = dim3((n + U - 1) / U, (m + T - 1) / T, 1);
  // Invoke CUDA kernel -----------------------------------------------------
  mysgemm<<<gridDim, blockDim>>>(m, n, k, A, B, C);
  // INSERT CODE HERE
}
