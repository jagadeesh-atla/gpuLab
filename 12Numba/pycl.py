import pyopencl as cl


def print_device_info():
    print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
    for platform in cl.get_platforms():
        print('='*60)
        print('Platform - Name:    ' + platform.name)
        print('Platform - vendor:  ' + platform.vendor)
        print('Platform - version: ' + platform.version)
        print('Platform - profile: ' + platform.profile)

        for device in platform.get_devices():
            print('\t' + '-'*52)
            print('\tDevice - Name: ' + device.name)
            print('\tDevice - Type: ' + cl.device_type.to_string(device.type))
            print(f'\tDevice - Max Clock Speed: {device.max_clock_frequency}')
            print(
                '\tDevice - Compute Units: {0}'.format(device.max_compute_units))
            print(
                '\tDevice - Local Memory: {0:.0f} KB'.format(device.local_mem_size / 1024.0))
            print(
                '\tDevice - constant Memory: {0:.0f} KB'.format(device.max_constant_buffer_size / 1024.0))
            print(
                '\tDevice - global Memory: {0:.0f} GB'.format(device.global_mem_size / (1024.0 * 1024.0 * 1024.0)))
            print('\tDevice - Max Buffer/Image Size: {0:.0f} MB'.format(
                device.max_mem_alloc_size / 1073741824.0))
            print(
                '\tDevice - max Work Group Size: {0:.0f}'.format(device.max_work_group_size))

    print('\n')


if __name__ == '__main__':
    print_device_info()
