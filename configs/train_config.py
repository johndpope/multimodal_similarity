"""
Default configurations for model training 
"""

from .base_config import BaseConfig
import argparse

class TrainConfig(BaseConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()

        self.parser.add_argument('--pretrained_model', type=str, default=None,
                help='absolute path of pretrained model')
        self.parser.add_argument('--feat', type=str, default='resnet',
                help='feature used: resnet | sensors')
        self.parser.add_argument('--network', type=str, default='tsn',
                help='Network used for sequence encoding: tsn | lstm')
        self.parser.add_argument('--metric', type=str, default='squaredeuclidean',
                help='Metric used to calculate distance: squaredeuclidean | euclidean | l1')
        self.parser.add_argument('--no_normalized', dest='normalized', action="store_false",
                help='Whether embeddings are normalized to unit vector')
        self.parser.set_defaults(normalized=True)
        self.parser.add_argument('--reverse', dest='reverse', action="store_true",
                help='Whether to reverse input sequence')
        self.parser.set_defaults(reverse=False)

        self.parser.add_argument('--num_threads', type=int, default=2,
                       help='number of threads for loading data in parallel')
        self.parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
        self.parser.add_argument('--max_epochs', type=int, default=5,
                       help='Max epochs')
        self.parser.add_argument('--sess_per_batch', type=int, default=3,
                       help='# of sessions per batch')
        self.parser.add_argument('--event_per_batch', type=int, default=1000,
                       help='# of event per batch')
        self.parser.add_argument('--triplet_per_batch', type=int, default=100,
                help='number of triplets per batch. Note: according to implemetation, actual amount may be larger than this number (by a constant number), also may be smaller than this number (short sessions)')
        self.parser.add_argument('--num_negative', type=int, default=3,
                       help='# of negative samples per anchor-positive pairs')
        self.parser.add_argument('--num_seg', type=int, default=3,
                       help='# of segment for a sequence')
        self.parser.add_argument('--emb_dim', type=int, default=256,
                       help='dimensionality of embedding')
        self.parser.add_argument('--triplet_select', type=str, default='random',
                help='methods for triplet selection: random | facenet |')
        self.parser.add_argument('--alpha', type=float, default=0.2,
                       help='margin for triplet loss')
        self.parser.add_argument('--lambda_l2', type=float, default=0.0,
                       help='L2 regularization')
        self.parser.add_argument('--keep_prob', type=float, default=1.0,
                help='Keep prob for dropout')
        self.parser.add_argument('--negative_epochs', type=int, default=0,
                       help='Start hard negative mining after this number of epochs')

        self.parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='initial learning rate')
        self.parser.add_argument('--static_epochs', type=int, default=1000,
                       help='number of epochs using the initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='ADAM',
                help='optimizer: ADAM | RMSPROP | MOMEMTUM | ADADELTA | SGD | ADAGRAD')

        self.parser.add_argument('--gpu', type=str, default=0,
                help='Set CUDA_VISIBLE_DEVICES')

