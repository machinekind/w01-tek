# A recognition hypothesis for one utterance.  Partials (final=false) are
# progressive rewrites of the same utterance_id; the final transcript
# replaces them all and is the only thing the router acts on.
std_msgs/Header header
string utterance_id
string text
bool final
float32 confidence      # engine-specific; mean avg_logprob for whisper
string language
