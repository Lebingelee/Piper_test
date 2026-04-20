from pyorbbecsdk import *

ctx = Context()
device_list = ctx.query_devices()
count = device_list.get_count()
print("device_count =", count)

for i in range(count):
    dev = device_list.get_device_by_index(i)
    info = dev.get_device_info()
    print("name =", info.get_name())
    print("serial =", info.get_serial_number())
