# One sentence of a spoken reply, flushed on punctuation so TTS can start
# before the LLM finishes.  seq orders sentences within a reply; final marks
# the last one.
std_msgs/Header header
string utterance_id     # the user utterance this reply answers
uint32 seq
string text
bool final
string source           # bielik | qwen | system
