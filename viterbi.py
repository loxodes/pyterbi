# jon klein
# kleinjt@ieee.org
# mit license

# viterbi algorithim, in pycuda and python

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
import numpy
import sys
import pp
import matplotlib.pyplot as plt

def main():
    trellises = 20 
    cores = range(1,4)
    times = [speedup_calc(16,16,512,trellises,c) for c in cores]
    print 'execution time', times

    
def benchmark_multitrellis():
    pass

def benchmark_singletrellis():
    trellises = 4;
    cores = 2;
    nobs = [pow(2,i) for i in range(4,12)];
    noutputs = 32;
    nstates = [pow(2,i) for i in range(1,7)]
    speedup = [[speedup_calc(o, noutputs, n, trellises, cores) for n in nstates] for o in nobs]
    f = plt.figure()

    plots = [plt.plot(nstates, s) for s in speedup]
    plt.legend(plots, nobs,title='observations',bbox_to_anchor=(1.10,1))

    plt.xlabel('number of states')
    plt.ylabel('speedup over host only implementation')
    plt.title('speedup of PyCUDA pyterbi over host only viterbi decoder\n32 outputs, variable number of states and observations lengths\nCPU: i5-2520M, GPU: NVS4200M')
    plt.grid(True)
    plt.savefig('speedup_graph.png')
    plt.show()

def speedup_calc(nobs, noutputs, nstates, ntrellises, hostcores):
    trellises = []
    for i in range(ntrellises):
        trellises.append(Trellis(nobs, noutputs, nstates))
    
    start = cuda.Event()
    end = cuda.Event()
   
    # benchmark host path with an arbitrary number of host cores
    start.record() 
    job_server = pp.Server(ncpus=hostcores)
   
    jobs = []
    for t in trellises:
        jobs.append(viterbi_ppjob(job_server, t))
   
    job_server.wait() 
    for i in range(len(trellises)):
        j = jobs[i];
        t = trellises[i];
        t.hostroute = j()
    
    
    end.record()
    end.synchronize()
    host_time = start.time_till(end) * 1e-3
    
    ''' 
    # benchmark cuda path
    start.record()
    route_cu = viterbi_cuda(obs, states, init_p, trans_p, emit_p, trellises)
    end.record()
    end.synchronize()
    cuda_time = start.time_till(end) * 1e-3

    # report on results
    if (numpy.array_equal(route_cu, route_host)):
        pass
    else:
        print 'host and cuda paths do *NOT* match!'
        pdb.set_trace()

    return (host_time/cuda_time)
'''
    return host_time

class Trellis:
    def __init__(self, nobs, noutputs, nstates):
        self.states = numpy.array(range(nstates))
        self.obs = numpy.array(numpy.random.randint(noutputs,size=nobs),dtype=numpy.uint16)
        
        self.init_p = numpy.random.rand(nstates)
        self.init_p = numpy.log(numpy.array((self.init_p/sum(self.init_p)),dtype=numpy.float32))
        
        self.trans_p = numpy.random.rand(nstates,nstates)
        self.trans_p = numpy.transpose(self.trans_p / sum(self.trans_p))
        self.trans_p = numpy.log(numpy.array(self.trans_p, dtype=numpy.float32))
    
        self.emit_p = numpy.random.rand(nstates,noutputs)
        self.emit_p = numpy.transpose(self.emit_p / sum(self.emit_p))
        self.emit_p = numpy.log(numpy.array(self.emit_p, dtype=numpy.float32))
        
        self.cudaroute = []
        self.hostroute = []
        self.refroute = []

def viterbi_host(obs, states, init_p, trans_p, emit_p):
    nobs = len(obs)
    nstates = len(states)
    path_p = numpy.zeros((nobs,nstates), dtype=numpy.float32)
    back = numpy.zeros((nobs,nstates), dtype=numpy.int32)

    # set inital probabilities and path
    path_p[0,:] = init_p + emit_p[obs[0]]
    back[0,:] = states

    for n in range(1, nobs):
        for m in states:
            p = emit_p[obs[n]][m] + trans_p[:,m] + path_p[n-1]
            back[n][m] = numpy.argmax(p)
            path_p[n][m] = numpy.amax(p)  

    route = viterbi_backtrace(nobs, path_p, back)
    
    return route, path_p, back

def viterbi_ppjob(server, tr):
    return server.submit(viterbi_host, (tr.obs, tr.states, tr.init_p, tr.trans_p, tr.emit_p,),(viterbi_backtrace,),("numpy",))

    

def viterbi_cuda(obs, states, init_p, trans_p, emit_p, trellises):
    

    nobs = len(obs)
    nstates = len(states)
    path_p = numpy.zeros((nobs,nstates,trellises), dtype=numpy.float32)
    back = numpy.zeros((nobs,nstates,trellises), dtype=numpy.int32)

    # set inital probabilities and path
    path_p[0,:] = init_p + emit_p[obs[0]]
    back[0,:] = states

    # allocate and copy arrays to global or constant memory
    emit_p_gpu = cuda.mem_alloc(emit_p.nbytes) 
    cuda.memcpy_htod(emit_p_gpu, emit_p)

    trans_p_gpu = cuda.mem_alloc(trans_p.nbytes) 
    cuda.memcpy_htod(trans_p_gpu, trans_p)
    
    obs_gpu = mod.get_global('obs')[0]
    cuda.memcpy_htod(obs_gpu, obs) 

    path_p_gpu = cuda.mem_alloc(path_p.nbytes)
    cuda.memcpy_htod(path_p_gpu, path_p)
    
    back_gpu = cuda.mem_alloc(back.nbytes)
    cuda.memcpy_htod(back_gpu, back)

    nstates_gpu = numpy.int32(nstates)
    nobs_gpu = numpy.int32(nobs)

    viterbi_cuda = mod.get_function("viterbi_cuda")

    viterbi_cuda(trans_p_gpu, emit_p_gpu, path_p_gpu, back_gpu, nstates_gpu, nobs_gpu, block=(nstates,1,1));

    cuda.memcpy_dtoh(back, back_gpu)
    cuda.memcpy_dtoh(path_p, path_p_gpu)
    
    route = viterbi_backtrace(nobs, path_p, back)
    
    return route, path_p, back

def viterbi_backtrace(nobs, path_p, back):
    route = numpy.zeros((nobs,1),dtype=numpy.int32)
    route[-1] = numpy.argmax(path_p[-1,:])

    for n in range(2,nobs+1):
        route[-n] = back[nobs-n+1,route[nobs-n+1]]
    return route;

mod = SourceModule("""
#include <stdio.h> 

#define MAX_OBS 4096 
#define MAX_STATES 64 
#define MAX_OUTS 64

__device__ __constant__ unsigned short obs[MAX_OBS];

__global__ void viterbi_cuda(float *trans_p, float *emit_p, float *path_p, int *back, int nstates, int nobs)
{
    const int tx = threadIdx.x;
    int i, j, ipmax;
    
    __shared__ float emit_p_s[MAX_OUTS * MAX_STATES];
    __shared__ float trans_p_s[MAX_STATES * MAX_STATES];
    __shared__ float path_p_s[MAX_STATES];
    
    for(i = 0; i < MAX_OUTS; i++) {
        emit_p_s[tx + i*nstates] = emit_p[tx + nstates * i];
    }

    for(i = 0; i < nstates; i++) {
        trans_p_s[tx + nstates * i] = trans_p[tx + nstates * i];
    }

    for(j = 1; j < nobs; j++) {
        path_p_s[tx] = path_p[(j-1)*nstates + tx];
        __syncthreads();
 
        float pmax = logf(0);
        float pt = 0; 
        ipmax = 0;

        for(i = 0; i < nstates; i++) {
            pt = emit_p_s[obs[j]*nstates+tx] + trans_p_s[i*nstates+tx] + path_p_s[i];
            if(pt > pmax) {
                ipmax = i;
                pmax = pt;
            }
        }
    
        path_p[j*nstates+tx] = pmax;
        back[j*nstates+tx] = ipmax;
        __syncthreads();
    }
}
""")

if __name__ == "__main__":
    main()
