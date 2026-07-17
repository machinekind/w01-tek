import fcntl, glob, os
for d in glob.glob("/sys/bus/usb/devices/*/idVendor"):
    if open(d).read().strip() == "0069":
        dev = os.path.dirname(d)
        bus = open(dev + "/busnum").read().strip().zfill(3)
        num = open(dev + "/devnum").read().strip().zfill(3)
        path = "/dev/bus/usb/%s/%s" % (bus, num)
        print("resetting", path)
        with open(path, "wb") as f:
            fcntl.ioctl(f, 0x5514)
        print("reset OK")
