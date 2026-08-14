# One look at how the computer running wojtek is doing.
#
# sysinfo_node fills the whole message inside one timer callback and stamps
# it once, so every field describes the same moment. On a host that is not a
# Raspberry Pi the Pi-only fields stay at zero.

std_msgs/Header header

# One entry per core, in core order.
float32[] cpu_percent
# Run queue length averaged over the last minute.
float32 load1
float32 cpu_freq_mhz
float32 soc_temp_c

# Raspberry Pi throttle status, decoded from the word vcgencmd reports.
# A "now" flag says what is happening at this moment. An "ever" flag stays
# set once it has happened since boot, so a brownout that lasted a
# millisecond is still visible half an hour later.
bool undervoltage_now
bool undervoltage_ever
bool throttled_now
bool throttled_ever
bool freq_capped_now
bool freq_capped_ever

uint64 mem_used_bytes
uint64 mem_total_bytes
uint64 swap_used_bytes
# Free space on the filesystem the rosbag is being written to.
uint64 disk_free_bytes

# Wifi traffic, averaged over the gap between this sample and the last one.
float32 wifi_rx_bytes_per_s
float32 wifi_tx_bytes_per_s
