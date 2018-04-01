"""
Default configurations for model evaluation 
"""

from .base_config import BaseConfig
import argparse

class EvalConfig(BaseConfig):
    def __init__(self):
        super(EvalConfig, self).__init__()

        self.parser.add_argument('--model_path', type=str, default=None,
                help='absolute path of pretrained model (including snapshot number')
        self.parser.add_argument('--feat', type=str, default='resnet',
                help='feature used')
        self.parser.add_argument('--seed', type=int, default=12345,
                       help='seed')
        self.parser.add_argument('--network', type=str, default='tsn',
                help='Network used for sequence encoding')
        self.parser.add_argument('--preprocess_func', type=str, default='mean',
                help='Preprocessing function for input, ignored when model is defined: mean | max')

        self.parser.add_argument('--num_seg', type=int, default=3,
                       help='# of segment for a sequence')
        self.parser.add_argument('--emb_dim', type=int, default=256,
                       help='dimensionality of embedding')

        self.parser.add_argument('--gpu', type=str, default=0,
                help='Set CUDA_VISIBLE_DEVICES')


