# coding=utf-8
import argparse
import logging
import random
import numpy as np
import torch
from datasets import load_datasets_and_vocabs
from models import TER_MCEE
from trainer import train

logger = logging.getLogger()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_path', type=str, default='./data', help='Dataset path.')
    parser.add_argument('--dataset_name', type=str, default='ChFinAnn',help='Choose ChFinAnn dataset.')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to store output data.')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory to store cache data.')
    parser.add_argument('--event_type_nums', type=int, default=5, help='Number of classes of event type.')
    parser.add_argument('--seed', type=int, default=2022, help='random seed for initialization')

    # Model parameters
    parser.add_argument('--token_embedding_dim', type=int, default=768, help='Dimension of token embedding')
    parser.add_argument('--sen_pos_embedding_dim', type=int, default=50, help='Dimension of sentence position embedding')
    parser.add_argument('--word_pos_embedding_dim', type=int, default=50, help='Dimension of word position embedding')
    parser.add_argument('--event_type_embedding_dim', type=int, default=50, help='Dimension of event type embedding')
    parser.add_argument('--event_embedding_dim', type=int, default=50, help='Dimension of event embedding')
    parser.add_argument('--pos_embedding_dim', type=int, default=50, help='Dimension of pos embeddings')
    parser.add_argument('--dep_embedding_dim', type=int, default=50, help='Dimension of dep embeddings')
    parser.add_argument('--train_event_num', type=int, default=34, help='Number of events contained in train set')
    parser.add_argument('--dev_event_num', type=int, default=15, help='Number of events contained in train set')
    parser.add_argument('--test_event_num', type=int, default=16, help='Number of events contained in test set')

    # MLP
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm.')
    parser.add_argument('--num_mlps', type=int, default=2, help='Number of mlps in the last of model.')
    parser.add_argument('--final_hidden_size', type=int, default=100, help='Hidden size of mlps.')

    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for embedding.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=15.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=20,
                        help="Log every X updates steps.")


    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    '''
    logger.info(vars(args))


def main():
    # Setup logging
    for h in logger.handlers:
        logger.removeHandler(h)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)

    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = 'cpu'
    args.device = 1
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    train_dataset,train_labels_weight,dev_dataset,dev_labels_weight,test_dataset,test_labels_weight,word_vocab, \
    pos_tag_vocab,dep_tag_vocab,sen_pos_tag_vocab,word_pos_tag_vocab,event_id_tag_vocab = load_datasets_and_vocabs(args)

    # Build Model
    model = TER_MCEE(args, pos_tag_vocab['len'], dep_tag_vocab['len'], sen_pos_tag_vocab['len'], word_pos_tag_vocab['len'],
                 event_id_tag_vocab['len'])

    model.to(args.device)

    # Train
    train(args,model,train_dataset,train_labels_weight,dev_dataset,dev_labels_weight,test_dataset,test_labels_weight)


if __name__ == "__main__":
    main()

