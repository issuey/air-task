REGISTRY = {}

from .rnn_agent import RNNAgent
from .maic_agent import MAICAgent
from .atten_rnn_agent import ATTRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY['maic'] = MAICAgent
REGISTRY["att_rnn"] = ATTRNNAgent
