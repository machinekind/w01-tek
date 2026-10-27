# One frame of mono PCM16 audio moving through the voice pipeline.
#
# On /wojtek/audio/mic (raw browser/mic stream) utterance_id is empty and
# end_of_utterance is always false.  On /wojtek/audio/speech the VAD sets
# utterance_id on every chunk of a segment; the closing chunk has
# end_of_utterance=true and carries the FULL utterance PCM (pre-roll
# included), so a final-only consumer needs no reassembly while a streaming
# consumer can use the live chunks and ignore the recap.
std_msgs/Header header
uint32 sample_rate
uint8 channels
uint64 seq              # per-publisher monotonic frame counter
string utterance_id
bool end_of_utterance
int16[] samples
