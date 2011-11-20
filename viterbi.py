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
    trellises = 10 
    cores = range(1,4)
    times = [speedup_calc(8,8,64,trellises,c) for c in cores]
    print 'speedup due to multicore', times

    
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
    run_hostviterbi(trellises, hostcores)
    end.record()
    end.synchronize()
    host_time = start.time_till(end) * 1e-3
   
    # benchmark host path with 1 host core, reference implementation
    start.record()
    run_hostviterbi(trellises, 1)
    end.record()
    end.synchronize()
    ref_time = start.time_till(end) * 1e-3

     
    # benchmark cuda path
    start.record()
    viterbi_cuda(trellises)
    end.record()
    end.synchronize()
    cuda_time = start.time_till(end) * 1e-3
    
    # report on results
    for t in trellises:
        if(t.checkroutes()):
            pass
        else:
            print 'host and cuda paths do *NOT* match!'
            #pdb.set_trace()

    return [ref_time/host_time, ref_time/cuda_time]

def run_hostviterbi(trellises, hostcores):
    job_server = pp.Server(ncpus=hostcores)
    jobs = []
    
    for t in trellises:
        jobs.append(t.get_ppjob(job_server))
    
    job_server.wait()
    
    for i in range(len(trellises)):
        j = jobs[i];
        t = trellises[i];
        t.routes.append(j())

# I'm trying to pretend this is a struct...
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
        
        self.routes = []
    
    def cuda_prep(self):
        # calculate sizes and create host-side path and back
        self.nstates = len(self.states)
        self.nobs = len(self.obs)
        self.path_p = numpy.zeros((self.nobs,self.nstates), dtype=numpy.float32)
        self.back = numpy.zeros((self.nobs,self.nstates), dtype=numpy.int32)

        self.path_p[0,:] = self.init_p + self.emit_p[self.obs[0]]
        self.back[0,:] = self.states
        

        # allocate and copy arrays to device global memory
        self.emit_p_gpu = cuda.mem_alloc(self.emit_p.nbytes) 
        cuda.memcpy_htod(self.emit_p_gpu, self.emit_p)

        self.trans_p_gpu = cuda.mem_alloc(self.trans_p.nbytes) 
        cuda.memcpy_htod(self.trans_p_gpu, self.trans_p)
    
        self.obs_gpu = cuda.mem_alloc(self.obs.nbytes)
        cuda.memcpy_htod(self.obs_gpu, self.obs) 

        self.path_p_gpu = cuda.mem_alloc(self.path_p.nbytes)
        cuda.memcpy_htod(self.path_p_gpu, self.path_p)
    
        self.back_gpu = cuda.mem_alloc(self.back.nbytes)
        cuda.memcpy_htod(self.back_gpu, self.back)

        self.nstates_gpu = numpy.int32(self.nstates)
        self.nobs_gpu = numpy.int32(self.nobs)

    def cuda_fetchresult(self):
            cuda.memcpy_dtoh(self.back, self.back_gpu)
            cuda.memcpy_dtoh(self.path_p, self.path_p_gpu)

    def checkroutes(self):
        for i in range(1,len(self.routes)):
            if(not numpy.array_equal(self.routes[i],self.routes[0])):
                return False
        return True

    def get_ppjob(self, server):
        return server.submit(viterbi_host, (self.obs, self.states, self.init_p, self.trans_p, self.emit_p,),(viterbi_backtrace,),("numpy",))


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
    return route

def viterbi_cuda(trellises):
    ref = cuda.Event()
    ref.record()

    stream, event = [], []
    marker_names = ['kernel_begin', 'kernel_end']

    viterbi_cuda = mod.get_function("viterbi_cuda")
    
    for t in trellises:
        stream.append(cuda.Stream())
        event.append(dict([(marker_names[l], cuda.Event()) for l in range(len(marker_names))]))

    for t in trellises:
        t.cuda_prep()

    for i in range(len(trellises)):
        event[i]['kernel_begin'].record(stream[i])        
        viterbi_cuda(t.obs_gpu, t.trans_p_gpu, t.emit_p_gpu, t.path_p_gpu, t.back_gpu, t.nstates_gpu, t.nobs_gpu, block=(t.nstates,1,1), stream=stream[i])

    for i in range(len(trellises)):
        event[i]['kernel_end'].record(stream[i])

    for t in trellises:
        t.cuda_fetchresult()
        t.routes.append(viterbi_backtrace(t.nobs, t.path_p, t.back))

def viterbi_backtrace(nobs, path_p, back):
    route = numpy.zeros((nobs,1),dtype=numpy.int32)
    route[-1] = numpy.argmax(path_p[-1,:])

    for n in range(2,nobs+1):
        route[-n] = back[nobs-n+1,route[nobs-n+1]]
    return route

mod = SourceModule("""
#include <stdio.h> 

#define MAX_OBS 4096 
#define MAX_STATES 64 
#define MAX_OUTS 64

__global__ void viterbi_cuda(short *obs, float *trans_p, float *emit_p, float *path_p, int *back, int nstates, int nobs)
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
