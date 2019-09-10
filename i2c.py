import io
import fcntl
from threading import Lock
from time

IOCTL_I2C_SLAVE = 0x0703


class I2C(object):
    '''i2c object that can be read and written to.
    Args:
        device (int): device address, typically given in hex format for your
            device datasheet
        bus (int): use `ls /dev/i2c*` to get all device addresses.
            "/dev/i2c-1" has a device address of 1
        address (int, optional): address of ioctl device slave. Default is
            IOCTL_I2C_SLAVE
    '''
    lock = Lock()

    def __init__(self, device, bus, address=IOCTL_I2C_SLAVE):
        self.fr = io.open("/dev/i2c-" + str(bus), "rb", buffering=0)
        self.fw = io.open("/dev/i2c-" + str(bus), "wb", buffering=0)

        # set device address
        fcntl.ioctl(self.fr, address, device)
        fcntl.ioctl(self.fw, address, device)

    def write(self, data: bytes):
        self.fw.write(data)

    def read(self, num: int):
        return self.fr.read(num)

    def close(self):
        self.fw.close()
        self.fr.close()

hx711 = I2C(0x48,0)  # Address = 0x48, I2C bus = 0

for x in range(3):
   raw_bytes=hx711.read(1)	# read 1 byte
   value = int.from_bytes(raw_bytes, byteorder='big')
   per_value=value/32767
   print(value,per_value)
   time.sleep_us(2)

print('finished')