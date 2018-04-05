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
        self.parser.add_argument('--network', type=str, default='tsn',
                help='Network used for sequence encoding')
        self.parser.add_argument('--preprocess_func', type=str, default='mean',
                help='Preprocessing function for input, ignored when model is defined: mean | max')

        self.parser.add_argument('--num_seg', type=int, default=3,
                       help='# of segment for a sequence')
        self.parser.add_argument('--emb_dim', type=int, default=256,
                       help='dimensionality of embedding')
        self.parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')

        self.parser.add_argument('--gpu', type=str, default=0,
                help='Set CUDA_VISIBLE_DEVICES')

        self.parser.add_argument('--no_normalized', dest='normalized', action="store_false",
                help='Whether embeddings are normalized to unit vector')
        self.parser.set_defaults(normalized=True)
        self.parser.add_argument('--reverse', dest='reverse', action="store_true",
                help='Whether to reverse input sequence')
        self.parser.set_defaults(reverse=False)

