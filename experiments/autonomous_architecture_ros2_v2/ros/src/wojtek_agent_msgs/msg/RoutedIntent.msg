# The router's verdict on one final transcript.  text is the transcript
# VERBATIM — downstream navigation must receive the user's exact words
# (lesson from #131: a paraphrase drops the route).
std_msgs/Header header
string utterance_id
string text
string intent           # chat | nav | visual | cancel | system
float32 confidence
