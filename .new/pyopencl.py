import pyopencl as cl

import numpy.linalg as la
import numpy

a = numpy.random.rand(256**3).astype(numpy.float32)

platform = cl.get_platforms()
my_gpus = platform[0].get_devices(device_type=cl.device_type.GPU)

ctx = cl.Context()
queue = cl.CommandQueue(ctx)

a_dev = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=a.nbytes)
cl._enqueue_write_buffer(queue, a_dev, a)


prg = cl.Program(ctx, """
                 __kernel void twice(__global float* a) {
                    a[get_global_id(0)] *= 2;
                 }
                """).build()
prg.twice(queue, a.shape, (1,), a_dev)

result = numpy.empty_like(a)

cl._enqueue_read_buffer(queue, a_dev, result).wait()

assert la.norm(result - 2*a) == 0
