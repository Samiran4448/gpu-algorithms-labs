#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__device__ uint64_t Intersect_Lin(const uint32_t *const edgeDst, size_t srcStart, size_t srcEnd, size_t dstStart, size_t dstEnd) {
  uint64_t TC = 0;
  if (srcStart < srcEnd && dstStart < dstEnd) {
    uint32_t W1 = edgeDst[srcStart];
    uint32_t W2 = edgeDst[dstStart];
    while (srcStart < srcEnd && dstStart < dstEnd) {
      if (W1 < W2) {
        W1 = edgeDst[++srcStart];
      } else if (W1 > W2) {
        W2 = edgeDst[++dstStart];
      } else if (W1 == W2) {
        W1 = edgeDst[++srcStart];
        W2 = edgeDst[++dstStart];
        ++TC;
      }
    }
    return TC;
  } else {
    return 0;
  }
}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  size_t edgeID = threadIdx.x + blockIdx.x * blockDim.x;
  if (edgeID < numEdges && numEdges > 0) {
    // Determine the source and destination node for the edge
    uint32_t src = edgeSrc[edgeID];
    uint32_t dst = edgeDst[edgeID];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    size_t srcStart = rowPtr[src];
    size_t srcEnd   = (numEdges > rowPtr[src + 1]) ? rowPtr[src + 1] : numEdges;

    size_t dstStart = rowPtr[dst];
    size_t dstEnd   = (numEdges > rowPtr[dst + 1]) ? rowPtr[dst + 1] : numEdges;
    // Determine how many elements of those two arrays are common

    triangleCounts[edgeID] = Intersect_Lin(edgeDst, srcStart, srcEnd, dstStart, dstEnd);
  }
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.

  uint64_t total = 0;

  //@@ calculate the number of blocks needed
  // dim3 dimGrid (ceil(number of non-zeros / dimBlock.x))

  //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
  // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
  // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
  pangolin::Vector<uint64_t> edge_counts = pangolin::Vector<uint64_t>(view.nnz(), 0);
  //@@ launch the linear search kernel here
  dim3 dimBlock(1024);
  if (mode == 1) {
    dim3 dimGrid((view.nnz() + dimBlock.x - 1) / dimBlock.x);
    // std::cout << view.nnz() << " view.nnz() test" << std::endl;
    // std::cout << dimGrid.x << " Grid dimensions test" << std::endl;
    kernel_tc<<<dimGrid, dimBlock>>>(edge_counts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();  
    // for (int i = 0; i < 10; i++) {
    //   printf("count %d\t", (int) edge_counts.data()[i]);
    //   printf("Edge source %d\t", view.row_ind()[i]);
    //   printf("Edge destination %d\n", view.col_ind()[i]);
    // }
    // printf("testing row_ptr array\n");
    // for (int i = 0; i < 10; i++) {
    //   printf("row ptr for %d\t to %d\n", view.row_ptr()[i], view.row_ptr()[i + 1]);
    // }

    // printf("row ptr from %d\t to %d\n", view.row_ptr()[1], view.row_ptr()[2]);
    // printf("Code reached here %d\n", __LINE__);

    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

  } else if (mode == 2) {

    //@@ launch the hybrid search kernel here
    // your_kernel_name_goes_here<<<dimGrid, dimBlock>>>(...)

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }

  
  //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
  // CPU reduction
  for (uint64_t i = 0; i < view.nnz(); ++i) {
    total += edge_counts.data()[i];
  }
  // printf("code reached here %d\n", __LINE__);

  // return total
  return total;

}