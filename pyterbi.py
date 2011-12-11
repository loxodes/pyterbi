# jon klein
# kleinjt@ieee.org
# mit license 
# 2011

# viterbi algorithm, in pycuda and python (pp parallelized for clusters or SMP)

# for 1024 observations and 64 states, pyterbi has roughly linear speedup with SMP cores
# roughly 100x speedup on a NVS4200M graphics card compared to a single core of an i5-2520M

# requires cuda compute capability 2.0 or higher
# tested on python 2.7.1, pp 1.6.1, pycuda 2011.1.3,

# obs       - observations                      [sample]
# states    - states                            [state] 
# init_p    - initial log probabilities         [state]
# trans_p   - transition log probabilities      [prev][current]
# emit_p    - output log probabilities          [emission][state]
# path_p    - initial and final path prob       [state]

import pdb 
import pycuda.tools
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy
import pp
import matplotlib.pyplot as plt

def main():
    trellises = 8 
    cores = 2
    times = speedup_calc(512,32,64,trellises,cores, False)
    print 'speedup due to parallelism:', times
#    benchmark_graphgen()

def benchmark_graphgen():
    trellises = 32 
    maxcores = 4
    nobs = [pow(2,i) for i in range(5,12)]
    noutputs = 32 
    nstates = [pow(2,i) for i in range(4,7)]
    speedups = [[speedup_calc(o, noutputs, n, trellises, maxcores) for n in nstates] for o in nobs]
    
    # sorry for the mess.. python list comprehension is too entertaining
    cuda_speedups = [[s[0] for s in n] for n in speedups]
    host_speedups = [[s[1] for s in n] for n in speedups]

    plots = [plt.plot(nstates, s) for s in cuda_speedups]
    plt.legend(plots, nobs,title='observations',bbox_to_anchor=(1.10,.6))
    plt.xlabel('number of states')
    plt.ylabel('speedup over host only implementation')
    plt.title('speedup of PyCUDA pyterbi over host only viterbi decoder\nCPU: i5-2520M, GPU: NVS4200M')
    
    plt.grid(True)
    plt.savefig('speedup_graph_cuda.png')

    plt.clf()
    plots = [plt.plot(nstates, s) for s in host_speedups]
    plt.legend(plots, nobs,title='observations',bbox_to_anchor=(1.10,.6))
    plt.xlabel('number of states')
    plt.ylabel('speedup over single process implementation')
    plt.title('speedup of multiple process parallized pyterbi over single process decoder\nCPU: i5-2520M')
    
    plt.grid(True)
    plt.savefig('speedup_graph_host.png')

def benchmark_host(trellises, cores, networked = False):
    start = cuda.Event()
    end = cuda.Event()

    start.record()
    run_hostviterbi(trellises, cores, networked)
    end.record()
    end.synchronize()
    
    ref_time = start.time_till(end) * 1e-3
    return ref_time

def benchmark_cuda(trellises):
    start = cuda.Event()
    end = cuda.Event()

    start.record()
    viterbi_cuda(trellises)
    end.record()
    end.synchronize()
    cuda_time = start.time_till(end) * 1e-3
    return cuda_time

def speedup_calc(nobs, noutputs, nstates, ntrellises, maxhostcores, networked = False):
    trellises = []

    for i in range(ntrellises):
        trellises.append(Trellis(nobs, noutputs, nstates))
    
    ref_time = benchmark_host(trellises, 1, False)
    cuda_time = benchmark_cuda(trellises)
    host_time = benchmark_host(trellises, maxhostcores, networked)
    
    for t in trellises:
        if(t.checkroutes()):
            pass
        else:
            print 'host and cuda paths do *NOT* match!'
            t = trellises[0]
            pdb.set_trace()

    return [ref_time/cuda_time, ref_time/host_time]

def run_hostviterbi(trellises, hostcores, networked=False):
    if not networked:
        job_server = pp.Server(ncpus=hostcores)
    else:
        ppservers=("*",)
        job_server = pp.Server(ppservers=ppservers, secret="pyterbi", ncpus=hostcores)
    jobs = []
    
    for t in trellises:
        jobs.append(t.get_ppjob(job_server))
    
    job_server.wait()

    for i in range(len(trellises)):
        j = jobs[i];
        t = trellises[i];
        t.routes.append(j())

    if networked:
        job_server.print_stats()
    
    job_server.destroy()


# I'm trying to pretend this is a struct...
class Trellis:
    def __init__(self, nobs, noutputs, nstates):
        self.states = numpy.array(range(nstates))
        self.obs = numpy.array(numpy.random.randint(noutputs,size=nobs),dtype=numpy.int16)
        
        self.init_p = numpy.random.rand(nstates)
        self.init_p = numpy.log(numpy.array((self.init_p/sum(self.init_p)),dtype=numpy.float32))
        
        self.trans_p = numpy.random.rand(nstates,nstates)
        self.trans_p = numpy.transpose(self.trans_p / sum(self.trans_p))
        self.trans_p = numpy.log(numpy.array(self.trans_p, dtype=numpy.float32))
    
        self.emit_p = numpy.random.rand(nstates,noutputs)
        self.emit_p = numpy.transpose(self.emit_p / sum(self.emit_p))
        self.emit_p = numpy.log(numpy.array(self.emit_p, dtype=numpy.float32))
        
        self.routes = []

    def checkroutes(self):
        if(len(self.routes) != 3):
            print 'trellis warning count, one of the runs may have failed!'
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
    back = numpy.zeros((nobs,nstates), dtype=numpy.int16)

    # set initial probabilities and path
    path_p[0,:] = init_p + emit_p[obs[0]]

    for n in range(1, nobs):
        for m in states:
            p = emit_p[obs[n]][m] + trans_p[:,m] + path_p[n-1]
            back[n][m] = numpy.argmax(p)
            path_p[n][m] = numpy.amax(p)

    route = viterbi_backtrace(nobs, path_p, back)
    return route

def viterbi_backtrace(nobs, path_p, back):
    route = numpy.zeros(nobs,dtype=numpy.int16)
    route[-1] = numpy.argmax(path_p[-1,:])

    for n in range(2,nobs+1):
        route[-n] = back[nobs-n+1,route[nobs-n+1]]
    return route


def viterbi_cuda(trellises):
    viterbi_cuda_gpu = mod.get_function("viterbi_cuda")
    viterbi_backtrace_gpu = mod.get_function("viterbi_cudabacktrace")
   
   
    # all trellises must be the same length/width
    noutputs = len(trellises[0].emit_p[:,0])
    nstates = len(trellises[0].states)
    nobs = len(trellises[0].obs)
    ntrellises = len(trellises)

    emit_size = noutputs * nstates * ntrellises
    trans_size = nstates * nstates * ntrellises
    path_size = nstates * ntrellises
    obs_size = nobs * ntrellises
    back_size = nobs * nstates * ntrellises

    nstates_gpu = numpy.int16(nstates)
    nobs_gpu = numpy.int16(nobs)
    nouts_gpu = numpy.int16(noutputs)

    path_p = numpy.zeros(path_size, dtype=numpy.float32)
    back = numpy.zeros(back_size, dtype=numpy.int16)
    emit_p = numpy.zeros(emit_size, dtype=numpy.float32)
    trans_p = numpy.zeros(trans_size, dtype=numpy.float32)
    obs = numpy.zeros(obs_size, dtype=numpy.int16)

    for i in range(ntrellises):
        emit_p[i*noutputs*nstates:(i+1)*noutputs*nstates] = trellises[i].emit_p.flatten()
        trans_p[i*nstates*nstates:(i+1)*nstates*nstates] = trellises[i].trans_p.flatten()
        obs[i*nobs:(i+1)*nobs] = trellises[i].obs
        path_p[i*nstates:(i+1)*nstates] = trellises[i].init_p + trellises[i].emit_p[trellises[i].obs[0]]
    

    # allocate and copy arrays to device global memory
    emit_p_gpu = cuda.mem_alloc(emit_p.nbytes) 
    cuda.memcpy_htod(emit_p_gpu, emit_p)

    trans_p_gpu = cuda.mem_alloc(trans_p.nbytes) 
    cuda.memcpy_htod(trans_p_gpu, trans_p)
    
    obs_gpu = cuda.mem_alloc(obs.nbytes)
    cuda.memcpy_htod(obs_gpu, obs) 

    path_p_gpu = cuda.mem_alloc(path_p.nbytes)
    cuda.memcpy_htod(path_p_gpu, path_p)
    
    back_gpu = cuda.mem_alloc(back.nbytes)
    
    viterbi_cuda_gpu(obs_gpu, trans_p_gpu, emit_p_gpu, path_p_gpu, back_gpu, nstates_gpu, nobs_gpu, nouts_gpu, block=(nstates,1,1), grid=(ntrellises,1))

    route = numpy.zeros(obs_size, dtype=numpy.int16)
    route_gpu = cuda.mem_alloc(route.nbytes)

    viterbi_backtrace_gpu(nobs_gpu, nstates_gpu, path_p_gpu, back_gpu, route_gpu, block=(ntrellises,1,1))     
    
    cuda.memcpy_dtoh(route, route_gpu)
    
    for i in range(ntrellises):
        t = trellises[i]
        t.routes.append(numpy.transpose(route[i*nobs:(i+1)*nobs]))

mod = SourceModule("""
#include <stdio.h> 

#define MAX_OBS 2048 
#define MAX_STATES 64 
#define MAX_OUTS 32 

__global__ void viterbi_cuda(short *obs, float *trans_p, float *emit_p, float *path_p, short *back, short nstates, short nobs, short nouts)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    short i, j, ipmax;
    
    __shared__ float emit_p_s[MAX_OUTS * MAX_STATES];
    __shared__ float trans_p_s[MAX_STATES * MAX_STATES];
    __shared__ float path_p_s[MAX_STATES];
    __shared__ float path_p_s_n[MAX_STATES];

    for(i = 0; i < nouts; i++) {
        emit_p_s[tx + i*nstates] = emit_p[tx + i*nstates + bx * nouts * nstates];
    }

    for(i = 0; i < nstates; i++) {
        trans_p_s[tx + nstates * i] = trans_p[tx + nstates * i + bx * nstates * nstates];
    }
    
    path_p_s_n[tx] = path_p[tx + bx*nstates];
    
    for(j = 1; j < nobs; j++) {
        path_p_s[tx] = path_p_s_n[tx];
        __syncthreads();
 
        float pmax = logf(0);
        float pt = 0; 
        ipmax = 0;

        for(i = 0; i < nstates; i++) {
            pt = emit_p_s[obs[nobs*bx+j]*nstates+tx] + trans_p_s[i*nstates+tx] + path_p_s[i];
            if(pt > pmax) {
                ipmax = i;
                pmax = pt;
            }
        }
    
        path_p_s_n[tx] = pmax;
        back[j*nstates+tx+bx*nstates*nobs] = ipmax;
        __syncthreads();
    }
    
    path_p[tx + bx*nstates] = path_p_s_n[tx];
    
}

__global__ void viterbi_cudabacktrace(short nobs, short nstates, float *path_p, short *back, short *route)
{
    const int tx = threadIdx.x;
    int i;

    float max_p = path_p[tx*nstates];
    short imax_p = 0;
    
    for(i=1; i<nstates; i++) {
        if(path_p[tx*nstates+i] > max_p) {
            max_p = path_p[tx*nstates+i];
            imax_p = i;
        }
    }

    route[tx*nobs + nobs-1] = imax_p; 
    
    for(i=nobs-2; i > -1; i--) {
        route[tx*nobs+i] = back[tx*nstates*nobs+(i+1)*nstates+route[tx*nobs + i+1]];
    }
}
""")

if __name__ == "__main__":
    main()
