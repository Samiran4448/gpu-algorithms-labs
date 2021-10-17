#include <cstdio>
#include <cstdlib>
#include <stdio.h>

#include "template.hu"

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE / WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

// __device__ atomicCAA(unsigned int* address, unsigned int value, unsigned int check){
//   unsigned int old = *address;
//   unsigned int assumed;
//   do
//   {
//     assumed = check;
//     old = atomicCAS(address, assumed, value + assumed)
//   } while (assumed != check);
// }

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
                                          unsigned int *nodeNeighbors,
                                          unsigned int *nodeVisited,
                                          unsigned int *currLevelNodes,
                                          unsigned int *nextLevelNodes,
                                          unsigned int *numCurrLevelNodes,
                                          unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the global queue
  unsigned int threadid_global = threadIdx.x + blockDim.x * blockIdx.x;
  for (unsigned int i = threadid_global; i < *numCurrLevelNodes; i += gridDim.x * blockDim.x) {
    unsigned int current = currLevelNodes[i];
    for (unsigned int j = nodePtrs[current]; j < nodePtrs[current + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];

      // set the neighbor to 1 and check if it was not already visited
      if (!atomicExch(&nodeVisited[neighbor], 1)) {
        unsigned int location    = atomicAdd(numNextLevelNodes, 1);
        nextLevelNodes[location] = neighbor;
      }
    }
  }
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
                                         unsigned int *nodeNeighbors,
                                         unsigned int *nodeVisited,
                                         unsigned int *currLevelNodes,
                                         unsigned int *nextLevelNodes,
                                         unsigned int *numCurrLevelNodes,
                                         unsigned int *numNextLevelNodes) {
  // INSERT KERNEL CODE HERE

  // Initialize shared memory queue
  __shared__ unsigned int bqNodes[BQ_CAPACITY];
  __shared__ unsigned int block_location, gq_location, bq_size;

  if (threadIdx.x == 0) {
    block_location = 0;
    gq_location    = 0;
    bq_size        = 0;
  }
  __syncthreads();
  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the block queue
  // If full, add it to the global queue
  unsigned int threadid_global = threadIdx.x + blockDim.x * blockIdx.x;
  for (unsigned int i = threadid_global; i < *numCurrLevelNodes; i += gridDim.x * blockDim.x) {
    unsigned int current = currLevelNodes[i];
    for (unsigned int j = nodePtrs[current]; j < nodePtrs[current + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];

      // set the neighbor to 1 and check if it was not already visited
      //V1
      if (!atomicExch(&nodeVisited[neighbor], 1)) {
        unsigned int location = atomicAdd(&block_location, 1);
        if (location < BQ_CAPACITY)
          bqNodes[location] = neighbor;
        else {
          location = atomicAdd(numNextLevelNodes, 1);
          nextLevelNodes[location] = neighbor;
        }
      }

      //V2
      // if (!atomicExch(&nodeVisited[neighbor], 1)) {
      //   unsigned int location = block_location;
      //   if (location < BQ_CAPACITY) {   //***THIS WILL SAVE SOME ATOMIC ADDITIONS BUT DOESN'T IMPROVE TIME FOR THE SMALL TESTCASE***// 
      //     location = atomicAdd(&block_location, 1);
      //     if(location<BQ_CAPACITY)
      //       bqNodes[location] = neighbor;
      //   } else {
      //     location    = atomicAdd(numNextLevelNodes, 1);
      //     nextLevelNodes[location] = neighbor;
      //   }
      // }

    }
  }
  __syncthreads();
  // Calculate space for block queue to go into global queue
  if (threadIdx.x == 0) {
    // no race condition on bq_size here
    bq_size     = (BQ_CAPACITY < block_location) ? BQ_CAPACITY : block_location;
    gq_location = atomicAdd(numNextLevelNodes, bq_size);
  }
  __syncthreads();
  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < bq_size; i += blockDim.x) {
    nextLevelNodes[gq_location + i] = bqNodes[i];
  }
}

__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
                                        unsigned int *nodeNeighbors,
                                        unsigned int *nodeVisited,
                                        unsigned int *currLevelNodes,
                                        unsigned int *nextLevelNodes,
                                        unsigned int *numCurrLevelNodes,
                                        unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE

  // This version uses one queue per warp

  // Initialize shared memory queue
  __shared__ unsigned int wqNodes[WQ_CAPACITY][NUM_WARPS];
  __shared__ unsigned int wq_location[NUM_WARPS];
  __shared__ unsigned int wq2bq_location[NUM_WARPS + 1]; // scan output
  __shared__ unsigned int wq_size[NUM_WARPS];

  __shared__ unsigned int bqNodes[BQ_CAPACITY];
  __shared__ unsigned int bq_location, bq_size, bq2gq_location;

  // Loop over all nodes in the curent level
  // Loop over all neighbors of the node
  // If the neighbor hasn't been visited yet
  // Add it to the warp queue
  // If full, add it to the block queue
  // If full, add it to the global queue
  unsigned int threadid_global = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int wx              = threadIdx.x / WARP_SIZE;
  unsigned int lx              = threadIdx.x % WARP_SIZE;

  unsigned int tx              = threadIdx.x % NUM_WARPS;   //coalesced SM accesses ? (check)

  // //Causes Divergence across all Warps
  // if(lx==0){
  //   wq_location[wx] = 0;
  // }

  if (threadIdx.x < NUM_WARPS) {
    wq_location[threadIdx.x] = 0;
    if(threadIdx.x==0){
      bq_location = 0;
    }
  }
  __syncthreads();
  // Calculate space for warp queue to go into block queue
  for (unsigned int i = threadid_global; i < *numCurrLevelNodes; i += gridDim.x * blockDim.x) {
    unsigned int current = currLevelNodes[i];
    for (unsigned int j = nodePtrs[current]; j < nodePtrs[current + 1]; j++) {
      unsigned int neighbor = nodeNeighbors[j];

      // set the neighbor to 1 and check if it was not already visited
      if (!atomicExch(&nodeVisited[neighbor], 1)) {
        unsigned int location = atomicAdd(&wq_location[tx], 1);
        if (location < WQ_CAPACITY) {
          wqNodes[location][tx] = neighbor;
        } else {
          location = atomicAdd(&bq_location, 1);
          if (location < BQ_CAPACITY) {
            bqNodes[location] = neighbor;
          } else {
            location = atomicAdd(numNextLevelNodes, 1);
            nextLevelNodes[location] = neighbor;
          }
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < NUM_WARPS) {
    wq_size[threadIdx.x] = (WQ_CAPACITY < wq_location[threadIdx.x]) ? WQ_CAPACITY : wq_location[threadIdx.x];
  }
  __syncthreads();
  
  // Store warp queue in block queue
  // If full, add it to the global queue

  //attempt 1 do Naive scan with 1 thread
  if(threadIdx.x==0){
    // bq_size = (BQ_CAPACITY < bq_location) ? BQ_CAPACITY : bq_location;

    wq2bq_location[0] = (BQ_CAPACITY < bq_location) ? BQ_CAPACITY : bq_location;
    for (int i = 0; i < NUM_WARPS; i++) {
      // unsigned int temp     = (WQ_CAPACITY < wq_location[i]) ? WQ_CAPACITY : wq_location[i];
      wq2bq_location[i + 1] = wq2bq_location[i] + wq_size[i];
    }

    bq_size = (BQ_CAPACITY < wq2bq_location[NUM_WARPS]) ? BQ_CAPACITY : wq2bq_location[NUM_WARPS];

    bq2gq_location = atomicAdd(numNextLevelNodes, bq_size);
  }
  __syncthreads();

  for (unsigned int i = lx; i < wq_size[wx]; i += WARP_SIZE) {
    unsigned int location = wq2bq_location[wx] + i;
    if(location<BQ_CAPACITY){
      bqNodes[location] = wqNodes[i][wx];
    }
    else{
      unsigned int location = atomicAdd(numNextLevelNodes, 1);
      nextLevelNodes[location] = wqNodes[i][wx];
    }
  }
  __syncthreads();


  // Calculate space for block queue to go into global queue
  // Saturate block queue counter
  // Calculate space for global queue

  // Store block queue in global queue
  for (unsigned int i = threadIdx.x; i < bq_size; i += blockDim.x) {
    nextLevelNodes[bq2gq_location + i] = bqNodes[i];
  }
}


/******************************************************************************
 Functions
*******************************************************************************/
// DON NOT MODIFY THESE FUNCTIONS!

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                        unsigned int *nodeVisited, unsigned int *currLevelNodes,
                        unsigned int *nextLevelNodes,
                        unsigned int *numCurrLevelNodes,
                        unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                       unsigned int *nodeVisited, unsigned int *currLevelNodes,
                       unsigned int *nextLevelNodes,
                       unsigned int *numCurrLevelNodes,
                       unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
                      unsigned int *nodeVisited, unsigned int *currLevelNodes,
                      unsigned int *nextLevelNodes,
                      unsigned int *numCurrLevelNodes,
                      unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel << <numBlocks, BLOCK_SIZE>>>
      (nodePtrs, nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
       numCurrLevelNodes, numNextLevelNodes);
}
