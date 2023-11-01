import pycuda.driver as drv

drv.init()

print(f"{drv.Device.count()} devices(s) found")

for ord in range(drv.Device.count()):
	dev = drv.Device(ord)
	print(f"Device #{ord}: {dev.name()}")
	print(f"\tCompute Capability: {dev.compute_capability()[0]}.{dev.compute_capability()[1]}")
	print(f"\tTotal Memory: {dev.total_memory() // 1024} KB")

