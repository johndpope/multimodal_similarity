"""
Basic configurations
"""
import os
import argparse

class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--name', type=str, default='debug',
                        help='name of this experiment')
        self.parser.add_argument('--silent_mode', action='store_true',
                help='Silent mode, no printing')
        self.parser.add_argument('--seed', type=int, default=12345,
                       help='seed')

        self.parser.add_argument('--ROOT', type=str, default='/home/xyang/projects/multimodal_similarity/',
                help='codes root path')
        self.parser.add_argument('--DATA_ROOT', type=str, default='/mnt/work/honda_100h/',
                help='data root path')

        self.parser.add_argument('--all_session', type=str, default='all',
                help='session id list for all sessions, e.g. 201704151140,201704141145, use "all" for all sessions, or input txt file name for specific sessions')
        self.parser.add_argument('--train_session', type=str, default='all',
                help='session id list for training, e.g. 201704151140,201704141145, use "all" for all sessions, or input txt file name for specific sessions')
        self.parser.add_argument('--val_session', type=str, default='all',
                help='session id list for validation, e.g. 201704151140,201704141145, use "all" for all sessions, or input txt file name for specific sessions')
        self.parser.add_argument('--test_session', type=str, default='all',
                help='session id list for test, e.g. 201704151140,201704141145, use "all" for all sessions, or input txt file name for specific sessions')

    def parse(self):
        args = self.parser.parse_args()

        args.feature_root = os.path.join(args.DATA_ROOT, 'features/')
        args.label_root = os.path.join(args.DATA_ROOT, 'labels/')
        args.result_root = os.path.join(args.DATA_ROOT, 'results/')
        args.tfrecords_root = os.path.join(args.DATA_ROOT, 'tfrecords2/')   # event-based
#        args.tfrecords_root = os.path.join(args.DATA_ROOT, 'tfrecords/')   # session-based

        # for multimodal input
        args.feat = args.feat.split(',')
        if len(args.feat) == 1:
            args.feat = args.feat[0]

        # define dimensions of features
        args.MAX_LENGTH_FRAMES = 90    # maximum number of frames kept for one event
        # for parsing tfrecords
        args.feat_dict = {'resnet': 98304, 'sensors':8}
        args.context_dict = {'label': 'int', 'length': 'int'}
        args.feat_dim = {'resnet': (8,8,1536), 'sensors':(8,)}

        if args.all_session == 'all':
            args.all_session = load_session_list(os.path.join(args.DATA_ROOT, 'all_session.txt'))
        elif args.all_session[-3:] == 'txt':
            args.all_session = load_session_list(os.path.join(args.DATA_ROOT, args.all_session))
        else:
            args.all_session = args.all_session.split(',')

        if args.train_session == 'all':
            args.train_session = load_session_list(os.path.join(args.DATA_ROOT, 'train_session.txt'))
        elif args.train_session[-3:] == 'txt':
            args.train_session = load_session_list(os.path.join(args.DATA_ROOT, args.train_session))
        else:
            args.train_session = args.train_session.split(',')

        if args.val_session == 'all':
            args.val_session = load_session_list(os.path.join(args.DATA_ROOT, 'val_session.txt'))
        elif args.val_session[-3:] == 'txt':
            args.val_session = load_session_list(os.path.join(args.DATA_ROOT, args.val_session))
        else:
            args.val_session = args.val_session.split(',')

        if args.test_session == 'all':
            args.test_session = load_session_list(os.path.join(args.DATA_ROOT, 'test_session.txt'))
        elif args.test_session[-3:] == 'txt':
            args.test_session = load_session_list(os.path.join(args.DATA_ROOT, args.test_session))
        else:
            args.test_session = args.test_session.split(',')
        return args

def load_session_list(path):
    with open(path, 'r') as fin:
        session_ids = fin.read().strip().split('\n')
    return session_ids
