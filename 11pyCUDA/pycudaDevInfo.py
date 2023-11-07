import pycuda.driver as drv

drv.init()

print(f"{drv.Device.count()} device(s) found.")

for ordinal in range(drv.Device.count()):
    dev = drv.Device(ordinal)
    print(f"Device #{ordinal} : {dev.name()}")
    print(
        f" Compute Capability: {'.'.join(str(x) for x in dev.compute_capability())}")
    print(f" Total Memory: {dev.total_memory() // (1024 * 1024)} MB")
