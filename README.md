# NN-optimisation-in-C++


To optimize the inference time, the direct concolution was replaced with im2col with GEMM(general matrix multiplication)
Implemented im2col to linearize inputs, convert 3D convol to 2D matrix mul.
Used OpenMP to do the thread level parallelism(TLP) accross output channels and GEMM tiles
Added SIMD to enable Data level parallelism(DLP) in inner loops
Also added thread local buffers for im2col operations.

These signifiiantly increased GFLOPs and throughput compared to the naive loops in the master branch while maintaining
the same accuracy 98.4%. 

The Performance metrics of this method is below:
 Performance metrics**********
 Final Accuracy: 98.4%
 GEMM tiles: TM=64 TN=64 TK=64
 OpenMP threads: 8
 Total Time:     678.311 ms
 Time per Image: 0.678311 ms
 Throughput:     1474.25 images/sec
 Performance:    1.2281 GFLOPs
***********************************

The Performance metrics of the naive loops is below:
Final Accuracy:98.4%
 Total Time:     1301.65 ms
 Time per Image: 1.30165 ms

by implmenting the above, 
  ~1.9x faster than the naive
  ~48% reduction in total time (from 1301ms to 678ms)
