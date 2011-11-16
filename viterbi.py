# jon klein
# kleinjt@ieee.org
# mit license

# viterbi algorithim, in pycuda and python
# course project for parallel computing class
# obs       - observations                      [sample]
# states    - states                            [state] 
# init_p    - initial log probabilities         [state]
# trans_p   - transition log probabilities      [prev][current]
# emit_p    - output log probabilities          [emission][state]

import pdb 
import pycuda.tools
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np 
from multiprocessing import Pool, Array, Process
import sys

BLOCK_SIZE = 16

def main():
    obs = np.array([1,1,1,0],dtype=np.int32)
    states = np.array([0,1])
    init_p = np.log(np.array([.9,.1],dtype=np.float32))
    trans_p = np.log(np.array([(.5,.5),(.1,.9)],dtype=np.float32))
    emit_p = np.log(np.array([(.8,.2),(.2,.8)], dtype=np.float32))
    route_host = viterbi_host(obs, states, init_p, trans_p, emit_p)
    route_cu = viterbi_cuda(obs, states, init_p, trans_p, emit_p)
    print 'cuda route:\n', route_cu
    print 'host route:\n', route_host

def viterbi_host(obs, states, init_p, trans_p, emit_p):
    # create negative infinity probability matrix
    nobs = len(obs)
    nstates = len(states)
    
    # track path through trellis
    path_p = np.zeros((nobs,nstates), dtype=np.float32)
    back = np.zeros((nobs,nstates), dtype=np.int32)

    # set inital probabilities and path
    path_p[0,:] = init_p + emit_p[obs[0]]
    back[0,:] = states

    for n in range(1, nobs):
        for m in states: # parallelize this:
            p = emit_p[obs[n]][m] + trans_p[:,m] + path_p[n-1]
            back[n][m] = np.argmax(p)
            path_p[n][m] = np.amax(p)  

    route = np.zeros((nobs,1),dtype=np.int32)
    route[-1] = np.argmax(path_p[-1,:])
    
    # backtrace to find likely route
    for n in range(2,nobs+1):
        route[-n] = back[nobs-n+1,route[nobs-n+1]]
    
    return route


def viterbi_cuda(obs, states, init_p, trans_p, emit_p):
    # create negative infinity probability matrix
    nobs = len(obs)
    nstates = len(states)
    
    # track path through trellis
    path_p = np.zeros((nobs,nstates), dtype=np.float32)
    back = np.zeros((nobs,nstates), dtype=np.int32)

    # set inital probabilities and path
    path_p[0,:] = init_p + emit_p[obs[0]]
    back[0,:] = states

    # copy constant arrays to device memory (emit_p, trans_p, obs)
    emit_p_gpu = cuda.mem_alloc(emit_p.nbytes) 
    cuda.memcpy_htod(emit_p_gpu, emit_p)

    trans_p_gpu = cuda.mem_alloc(trans_p.nbytes) 
    cuda.memcpy_htod(trans_p_gpu, trans_p)
    
    obs_gpu = cuda.mem_alloc(obs.nbytes) 
    cuda.memcpy_htod(obs_gpu, obs) 

    path_p_gpu = cuda.mem_alloc(path_p.nbytes)
    cuda.memcpy_htod(path_p_gpu, path_p)
    
    back_gpu = cuda.mem_alloc(back.nbytes)
    cuda.memcpy_htod(back_gpu, back)

    viterbi_step_func = mod.get_function("viterbi_cuda_step")

    nstates_gpu  = np.int32(len(states))
    
    # calculate viterbi steps
    for n in np.arange(1, nobs, dtype=np.uint32):
        for m in states:
            viterbi_step_func(obs_gpu, trans_p_gpu, emit_p_gpu, path_p_gpu, back_gpu, n, nstates_gpu, block=(nstates,1,1));

    cuda.memcpy_dtoh(back, back_gpu)
    cuda.memcpy_dtoh(path_p, path_p_gpu);
    route = np.zeros((nobs,1),dtype=np.int32)
    route[-1] = np.argmax(path_p[-1,:])
  
    # backtrace to find likely route
    for n in range(2,nobs+1):
        route[-n] = back[nobs-n+1,route[nobs-n+1]]
    
    return route

mod = SourceModule("""
#include <stdio.h> 

__global__ void viterbi_cuda_step(int *obs, float *trans_p, float *emit_p, float *path_p, int *back, int tick, int nstates)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y; 
    
    const int bx = blockIdx.x; 
    const int by = blockIdx.y;

    int i, ipmax;

    float pmax = logf(0);
    float pt = 0;
    ipmax = 0;

    for(i = 0; i < nstates; i++)
    {
        pt = emit_p[obs[tick]+tx*nstates] + trans_p[i*nstates+tx] + path_p[(tick-1)*nstates+i];
        if(pt > pmax) {
            ipmax = i;
            pmax = pt;
        }
    }
    
    __syncthreads();
    
    path_p[tick*nstates+tx] = pmax;
    back[tick*nstates+tx] = ipmax;
    
}
""")

if __name__ == "__main__":
    main()
