#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#define BLOCK_SIZE 512
#include "template.hu"


//Linear Intersection Function
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

// Binary Search Intersection Function
__device__ uint64_t Bin_Search(const uint32_t needle, const uint32_t *const edgeDst, const size_t heystackStart, const size_t heystackEnd) {
  size_t l = heystackStart;
  size_t r = heystackEnd - l;
  if (r >= 1) {
    size_t mid = l + (r - 1) / 2;
    if (edgeDst[mid] == needle)
      return (uint64_t) 1;
    else if (edgeDst[mid] > needle) {
      return Bin_Search(needle, edgeDst, heystackStart, /*(mid > heystackStart) ? */mid /*: heystackStart*/);
    } else
      return Bin_Search(needle, edgeDst, (mid + 1 < heystackEnd) ? mid + 1 : heystackEnd, heystackEnd);
  } 
  /*else if (edgeDst[heystackStart] == needle) {
    return (uint64_t) 1;
  }*/
  else 
    return (uint64_t) 0;
}

__device__ uint64_t Intersect_Bin(
    const uint32_t *const edgeDst, size_t srcStart, size_t srcEnd, const size_t dstStart, const size_t dstEnd, size_t edgeID) {
  uint64_t TC = 0;
  // find all nodes in src list in dst list
  if (srcStart < srcEnd && dstStart < dstEnd) {
    for (uint32_t i = srcStart; i < srcEnd; i++) {
      // if (edgeID == 106) {
      //   printf("%d, \t", (int) edgeDst[i]);
      // }
      uint32_t needle = edgeDst[i];
      TC += Bin_Search(needle, edgeDst, dstStart, dstEnd);
    }
    // if (edgeID == 106) {
    //   printf("\n");
    //   printf("Count here: %d\n", (Int) TC);
    //   for (uint32_t i = dstStart; i < dstEnd; i++) {
    //     printf("%d, \t", (int) edgeDst[i]);
    //   }
    //   printf("\n");
    // }
    return TC;
  } else
    return TC;
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

__global__ static void kernel_tc_bin(uint64_t *__restrict__ triangleCounts, //!< per-edge triangle counts
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
    size_t srcSize  = srcEnd - srcStart;
    size_t dstStart = rowPtr[dst];
    size_t dstEnd   = (numEdges > rowPtr[dst + 1]) ? rowPtr[dst + 1] : numEdges;
    size_t dstSize  = dstEnd - dstStart;

    size_t max = (srcSize > dstSize) ? srcSize : dstSize;
    size_t min = (srcSize < dstSize) ? srcSize : dstSize;

    if (min >= max / __log2f(max)) {
      // Determine how many elements of those two arrays are common
      if (srcSize <= dstSize) {
        triangleCounts[edgeID] = Intersect_Bin(edgeDst, srcStart, srcEnd, dstStart, dstEnd, edgeID);
      } else {
        triangleCounts[edgeID] = Intersect_Bin(edgeDst, dstStart, dstEnd, srcStart, srcEnd, edgeID);
      }
    } 
    else {
      triangleCounts[edgeID] = Intersect_Lin(edgeDst, srcStart, srcEnd, dstStart, dstEnd);
    }
  }
}

__global__ void GPU_total(uint64_t *inputList, uint64_t *outputList, const size_t len) {
      __shared__ uint64_t sdata[BLOCK_SIZE];
      unsigned int tid = threadIdx.x;
      unsigned int i   = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
      sdata[tid]              = 0;
      // sdata[tid + blockDim.x] = 0;
      __syncthreads();
      if (i < len) {
        sdata[tid] = inputList[i];
        if (i + blockDim.x < len) {
          sdata[tid] += inputList[i + blockDim.x];
        }
      }
      __syncthreads();
      for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
          sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
      }
      if (tid == 0) 
        outputList[blockIdx.x] = sdata[0];
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
  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid((view.nnz() + dimBlock.x - 1) / dimBlock.x);
  if (mode == 1) {
    
    // std::cout << view.nnz() << " view.nnz() test" << std::endl;
    // std::cout << dimGrid.x << " Grid dimensions test" << std::endl;
    kernel_tc<<<dimGrid, dimBlock>>>(edge_counts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();

    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count

  } else if (mode == 2) {

    //@@ launch the hybrid search kernel here
    kernel_tc_bin<<<dimGrid, dimBlock>>>(edge_counts.data(), view.row_ind(), view.col_ind(), view.row_ptr(), view.nnz());
    cudaDeviceSynchronize();

  } else {
    assert("Unexpected mode");
    return uint64_t(-1);
  }
  // for (int i = 1000; i < 1200; i++) {
  //   std::cout << edge_counts.data()[i] << "\t";
  //   if ((i+1) % 10 == 0)
  //     std::cout << std::endl;
  // }

    //@@ do a global reduction (on CPU or GPU) to produce the final triangle count
    // CPU reduction 
    // for (uint64_t i = 0; i < view.nnz(); ++i) {
    //   total += edge_counts.data()[i];
    // }
  //GPU reduction
  pangolin::Vector<uint64_t> block_edge_counts = pangolin::Vector<uint64_t>(dimGrid.x, 0);
  GPU_total<<<dimGrid, dimBlock>>>(edge_counts.data(), block_edge_counts.data(), view.nnz());
  cudaDeviceSynchronize();
  for (int i = 0; i < dimGrid.x; i++){
    total += block_edge_counts.data()[i];
  }
    // printf("code reached here %d\n", __LINE__);
  return total;
}