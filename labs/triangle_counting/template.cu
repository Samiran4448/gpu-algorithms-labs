#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

__device__ uint64_t Intersect(const uint32_t *const edgeDst, size_t srcStart, size_t srcEnd, size_t dstStart, size_t dstEnd){

}

__global__ static void kernel_tc(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
                                 const uint32_t *const edgeSrc,         //!< node ids for edge srcs
                                 const uint32_t *const edgeDst,         //!< node ids for edge dsts
                                 const uint32_t *const rowPtr,          //!< source node offsets in edgeDst
                                 const size_t numEdges                  //!< how many edges to count triangles for
) {
  size_t edgeID = threadIdx.x + blockIdx.x * blockDim.x;
  if (edgeID < numEdges) {
    // Determine the source and destination node for the edge
    uint32_t src = edgeSrc[edgeID];
    uint32_t dst = edgeDst[edgeID];

    // Use the row pointer array to determine the start and end of the neighbor list in the column index array
    size_t srcStart = rowptr[src];
    size_t srcEnd   = rowptr[src+1];

    size_t dstStart = rowptr[dst];
    size_t dstEnd   = rowptr[dst + 1];
    // Determine how many elements of those two arrays are common

    triangleCounts[edgeID] += Intersect(edgeDst, srcStart, srcEnd, dstStart, dstEnd);
  }
}

uint64_t count_triangles(const pangolin::COOView<uint32_t> view, const int mode) {
  if (mode == 1) {

    // REQUIRED

    //@@ create a pangolin::Vector (uint64_t) to hold per-edge triangle counts
    // Pangolin is backed by CUDA so you do not need to explicitly copy data between host and device.
    // You may find pangolin::Vector::data() function useful to get a pointer for your kernel to use.
    pangolin::Vector<uint64_t> edge_counts;
    //@@ launch the linear search kernel here
    dim3 dimBlock(512);
    dim3 dimGrid(ceil(view.nnz() / dimBlock.x));
    std::cout << view.nnz() << " view.nnz() test\n" << std::endl;
    std::cout << dimGrid.x << " Grid dimensions test\n" << std::endl;
    kernel_tc<<<dimGrid, dimBlock>>>(edge_counts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());

    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
    //CPU reduction
    for (int i = 0;i<view.nnz(),i++){
      total += edge_counts[i];
    }

      return total;

  } else if (2 == mode) {

    // OPTIONAL. See README for more details

    uint64_t total = 0;
    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

    return total;
  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
}
