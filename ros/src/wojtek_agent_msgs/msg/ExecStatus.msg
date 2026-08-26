# World -> agent heartbeat: everything the navigation controllers poll.
# One message instead of three topics so active/blocked/resets are always
# read as a consistent snapshot (the demo read them in one event loop tick).
std_msgs/Header header
float32 x
float32 y
float32 yaw
bool active          # executor is driving a command right now
uint32 blocked       # monotonic no-progress counter (midlevel stall aborts)
uint32 resets        # monotonic fall-reset counter
float64 sim_time     # world clock, seconds; feeds PoseHistory timestamps
string exec_json     # executor.status() dict as JSON, for the status tool
uint64 cmd_seq       # highest accepted WorldCommand seq REFLECTED in this
                     # snapshot -- the proxy holds executor.active high until
                     # status catches up with its last ack (in-process code
                     # saw submit() flip active synchronously; topics don't)
