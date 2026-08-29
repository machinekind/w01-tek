# Reply to a WorldCmd, correlated by req_seq.
std_msgs/Header header
uint64 req_seq
bool ok
string error
string command
uint64 cmd_seq   # accepted-command sequence (see ExecStatus.cmd_seq)
