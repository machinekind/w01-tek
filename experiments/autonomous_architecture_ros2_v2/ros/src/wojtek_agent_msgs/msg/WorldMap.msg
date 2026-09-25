# Agent-built occupancy grid, streamed world -> agent at a low rate.
# Mirrors wojtek_eval.mapping.OnlineMap so the agent side can rebuild a real
# OnlineMap instance and keep map_image()/frontier math unchanged.
std_msgs/Header header
float32 res           # meters per cell
float64[2] origin     # world (x, y) of cell (0, 0)
uint32[2] shape       # (rows, cols)
uint8[] state         # rows*cols row-major, {0 unknown, 1 free, 2 occupied}
float32[] trail       # interleaved x,y of the recent pose trail
float32 ego_fovy_deg  # ego camera vertical FOV; search bearing math needs it
