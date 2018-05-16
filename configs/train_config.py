"""
Default configurations for model training 
"""

from .base_config import BaseConfig
import argparse

class TrainConfig(BaseConfig):
    def __init__(self):
        super(TrainConfig, self).__init__()

        self.parser.add_argument('--model_path', type=str, default=None,
                help='absolute path of pretrained model')
        self.parser.add_argument('--sensors_path', type=str, default=None,
                help='absolute path of pretrained sensors model')
        self.parser.add_argument('--segment_path', type=str, default=None,
                help='absolute path of pretrained segment model')
        self.parser.add_argument('--feat', type=str, default='resnet',
                help='feature used: resnet | sensors')
        self.parser.add_argument('--network', type=str, default='tsn',
                help='Network used for sequence encoding: tsn | lstm | rtsn | convtsn | convrtsn')
        self.parser.add_argument('--metric', type=str, default='squaredeuclidean',
                help='Metric used to calculate distance: squaredeuclidean | euclidean | l1')
        self.parser.add_argument('--no_normalized', dest='normalized', action="store_false",
                help='Whether embeddings are normalized to unit vector')
        self.parser.set_defaults(normalized=True)
        self.parser.add_argument('--reverse', dest='reverse', action="store_true",
                help='Whether to reverse input sequence for seq2seq')
        self.parser.set_defaults(reverse=False)
        self.parser.add_argument('--no_soft', dest='no_soft', action="store_true",
                help='Whether to use softplus')
        self.parser.set_defaults(no_soft=False)
        self.parser.add_argument('--no_joint', dest='no_joint', action="store_true",
                help='Whether to use joint optimization')
        self.parser.set_defaults(no_soft=False)
        self.parser.add_argument('--weighted', dest='weighted', action="store_true",
                help='Whether to use weighted triplet loss')
        self.parser.set_defaults(reverse=True)

        self.parser.add_argument('--label_num', type=int, default=93,
                       help='number of sessions with labels used for training')
        self.parser.add_argument('--task', type=str, default="supervised",
                help='training task: supervised | semi-supervised | zero-shot')

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
        self.parser.add_argument('--n_h', type=int, default=8,
                       help='height of the feature map')
        self.parser.add_argument('--n_w', type=int, default=8,
                       help='width of the feature map')
        self.parser.add_argument('--n_C', type=int, default=20,
                       help='number of output channels')
        self.parser.add_argument('--n_input', type=int, default=1536,
                       help='dim of input')
        self.parser.add_argument('--triplet_select', type=str, default='random',
                help='methods for triplet selection: random | facenet |')
        self.parser.add_argument('--multimodal_select', type=str, default='random',
                help='methods for multimodal selection: random | confidence |')
        self.parser.add_argument('--alpha', type=float, default=0.2,
                       help='margin for triplet loss')
        self.parser.add_argument('--lambda_l2', type=float, default=0.0,
                       help='L2 regularization')
        self.parser.add_argument('--lambda_ver', type=float, default=0.0,
                       help='if lambda_ver > 0, then multitask learning (verification loss) is used, and lambda_ver to balance its contribution')
        self.parser.add_argument('--lambda_multimodal', type=float, default=0.0,
                       help='lambda for multimodal weighted_metric_loss')
        self.parser.add_argument('--keep_prob', type=float, default=1.0,
                help='Keep prob for dropout')
        self.parser.add_argument('--negative_epochs', type=int, default=0,
                       help='Start hard negative mining after this number of epochs')
        self.parser.add_argument('--multimodal_epochs', type=int, default=0,
                       help='When to start multimodal joint optimization')

        self.parser.add_argument('--learning_rate', type=float, default=0.05,
                       help='initial learning rate')
        self.parser.add_argument('--static_epochs', type=int, default=1000,
                       help='number of epochs using the initial learning rate')
        self.parser.add_argument('--optimizer', type=str, default='ADAM',
                help='optimizer: ADAM | RMSPROP | MOMEMTUM | ADADELTA | SGD | ADAGRAD')

        self.parser.add_argument('--gpu', type=str, default=0,
                help='Set CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--label_type', type=str, default='goal',
                help='label_type: goal | stimuli')

        self.parser.add_argument('--loss', type=str, default='triplet',
                help='loss used: triplet | lifted | cluster | npairs')
