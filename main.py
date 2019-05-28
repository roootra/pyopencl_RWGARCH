# coding=utf-8
import numpy as np
import pyopencl as cl
import os
import arch
from pyopencl import cltypes
import pickle

# stub
if __name__ != "__main__":
    print("This script is stand-alone only and cannot be loaded as a Python module.")

'''
Used on Macbook Pro 2015 15' + macOS 10.14.4
Intel Core i7 4870HQ + AMD Radeon R9 M370X

kernel compilation fails if Intel device is chosen 
(looks like it is Apple OpenCL compiler issue)

Vector of random values is computed on CPU due to lack of RNG in OpenCL

randRets buffer has to be allocated in host memory, if randRets ndarray's size is greater than ~1 GB
due to VRAM limits (it is impossible to get amount of free VRAM on Apple platform and implemented OpenCL ext.)
'''

# user variables
nPer = 300
nSimul = 100000
p = 1
q = 1
directory = os.getcwd()
device = 2  # Intel CPU = 0, Intel Iris Pro = 1, AMD R9 M370X = 2

# loading data
with open(directory + "/pricesData.pkl", "rb") as fil:
    prices = pickle.load(fil)

# setting a context and command queue
platform = cl.get_platforms()[0]
device = platform.get_devices()[device]
ctx = cl.Context(devices=[device])
queue = cl.CommandQueue(ctx, device, cl.command_queue_properties.PROFILING_ENABLE)

# fulfiling memory allocation
randRets = np.random.rand(nPer * nSimul).astype(np.float32)  # data
print(randRets.nbytes, "bytes are allocated for simulated random returns.")
randRetsBuffer = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=randRets) # randRets
sample = np.array(prices["DJI"].loc[:, "Returns"])
sampleBuffer = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=sample)  # sampleBuff
resultBuffer = cl.Buffer(ctx, flags=cl.mem_flags.WRITE_ONLY, size=randRets.nbytes)  # resultBuf

# building a kernel
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'
txtKernelProg = open(directory + "/garchrwKernel.cl", "r").read()
prg = cl.Program(ctx, txtKernelProg).build()

# fitting a model
garchModel = arch.arch_model(prices["DJI"].loc[:, "Returns"], vol='Garch', p=p, o=0, q=q, dist='Normal')
garchModelFit = garchModel.fit()
print(garchModelFit.summary())
rwMean = garchModelFit.params[0]
garchMean = garchModelFit.params[1]
garchG = garchModelFit.params[2:(p+2)]
garchAR = garchModelFit.params[(p+2):(p+2+q)]
garchGBuffer = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(garchG))
garchARBuffer = cl.Buffer(ctx, flags=cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=np.array(garchAR))
print("Starting an execution of kernel routine...")
print("Device: " + device.name, ".\nGlobal memory: ",
      device.global_mem_size, " bytes.\nLocal memory: ",
      device.local_mem_size, " bytes.\n", sep="")
# executing opelcl kernel routine
prg.garchrw(queue, randRets.shape, None,
            resultBuffer,
            np.int32(nSimul),
            np.int32(nPer),
            randRetsBuffer,
            cltypes.float(rwMean),
            np.int32(len(garchG)),
            np.int32(len(garchAR)),
            cltypes.float(garchMean),
            garchGBuffer,
            garchARBuffer,
            sampleBuffer,
            np.int32(len(sample))
            )
print("Execution in progress...!")
result = np.empty_like(randRets)
cl.enqueue_copy(queue, result, resultBuffer)
print("Done!")

resultDF = result.reshape((nPer, nSimul), order="F")
resultSorted = resultDF.copy()

for col in range(0, resultDF.shape[1]):
    resultSorted[:, col] = np.sort(resultDF[:, col], axis=0, kind="heapsort")

with(open(directory + "/rwGarchResults.gz", "x")) as fil:
    np.savetxt(fil, resultDF, delimiter=",")
