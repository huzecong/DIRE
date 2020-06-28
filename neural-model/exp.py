"""
Variable renaming

Usage:
    exp.py train [options] CONFIG_FILE
    exp.py test [options] MODEL_FILE TEST_DATA_FILE

Options:
    -h --help                                   Show this screen
    --compressed                                Load dataset in compressed format
    --cuda                                      Use GPU
    --debug                                     Debug mode
    --seed=<int>                                Seed [default: 0]
    --work-dir=<dir>                            work dir [default: data/exp_runs/]
    --extra-config=<str>                        extra config [default: {}]
    --ensemble                                  Use ensemble
    --save-to=<str>                             Save decode results to path
"""
import pickle
import random
import time
from typing import List, Tuple, Dict, Iterable
import sys
import numpy as np
import os
import json
import _jsonnet
import pprint
from docopt import docopt
from tqdm import tqdm
import psutil, gc

import torch

from model.ensemble_model import EnsembleModel
from model.simple_decoder import SimpleDecoder
from model.graph_encoder import GraphASTEncoder
from model.gnn import AdjacencyList, GatedGraphNeuralNetwork
from model.model import RenamingModel
from utils import nn_util, util
from utils.ast import AbstractSyntaxTree
from utils.data_compress import CompressedDataset
from utils.dataset import Dataset, Example
from utils.evaluation import Evaluator
from utils.vocab import Vocab, VocabEntry


import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# os.environ['GPU_DEBUG'] = '2'
# from gpu_profile import gpu_profile


def log(msg: str) -> None:
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{time_str}] {msg}", flush=True, file=sys.stderr)


def train(args):
    work_dir = args['--work-dir']
    config = json.loads(_jsonnet.evaluate_file(args['CONFIG_FILE']))
    config['work_dir'] = work_dir

    if not os.path.exists(work_dir):
        print(f'creating work dir [{work_dir}]', file=sys.stderr)
        os.makedirs(work_dir)

    if args['--extra-config']:
        extra_config = args['--extra-config']
        extra_config = json.loads(extra_config)
        config = util.update(config, extra_config)

    json.dump(config, open(os.path.join(work_dir, 'config.json'), 'w'), indent=2)

    model = RenamingModel.build(config)
    config = model.config
    model.train()

    if args['--cuda']:
        model = model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    nn_util.glorot_init(params)

    # set the padding index for embedding layers to zeros
    # model.encoder.var_node_name_embedding.weight[0].fill_(0.)

    if args['--compressed']:
        train_set = CompressedDataset(config['data']['train_file'], random_seed=int(args['--seed']))
    else:
        train_set = Dataset(config['data']['train_file'])
    dev_set = Dataset(config['data']['dev_file'])
    batch_size = config['train']['batch_size']

    print(f'Training set size {len(train_set)}, dev set size {len(dev_set)}', file=sys.stderr)

    # training loop
    train_iter = epoch = cum_examples = 0
    log_every = config['train']['log_every']
    evaluate_every_nepoch = config['train']['evaluate_every_nepoch']
    evaluate_every_niter = config['train'].get('evaluate_every_niter', 0)
    max_epoch = config['train']['max_epoch']
    max_patience = config['train']['patience']
    cum_loss = 0.
    patience = 0.
    t_log = time.time()

    def validate():
        nonlocal patience
        log(f'[Learner] Perform evaluation')
        t1 = time.time()
        # ppl = Evaluator.evaluate_ppl(model, dev_set, config, predicate=lambda e: not e['function_body_in_train'])
        eval_results, decode_results = Evaluator.decode_and_evaluate(model, dev_set, config, return_results=True)

        save_to = (args['--save-to'] if args['--save-to']
                   else args['MODEL_FILE'] + f".{config['data']['dev_file'].split('/')[-1]}.decode_results.bin")
        print(f'Save decode results to {save_to}', file=sys.stderr)
        pickle.dump(decode_results, open(save_to, 'wb'))

        # print(f'[Learner] Evaluation result ppl={ppl} (took {time.time() - t1}s)', file=sys.stderr)
        log(f'[Learner] Evaluation result {eval_results} (took {time.time() - t1}s)')
        dev_metric = eval_results['func_body_not_in_train_acc']['accuracy']
        # dev_metric = -ppl
        if len(history_accs) == 0 or dev_metric > max(history_accs):
            patience = 0
            model_save_path = os.path.join(work_dir, f'model.bin')
            model.save(model_save_path)
            log(f'[Learner] Saved currently the best model to {model_save_path}')
        else:
            patience += 1
            if patience == max_patience:
                log(f'[Learner] Reached max patience {max_patience}, exiting...')
                patience = 0
                exit()

        history_accs.append(dev_metric)

    history_accs = []
    while True:
        # load training dataset, which is a collection of ASTs and maps of gold-standard renamings
        train_set_iter = train_set.batch_iterator(batch_size=batch_size,
                                                  return_examples=False,
                                                  config=config, progress=True, train=True,
                                                  num_readers=config['train']['num_readers'],
                                                  num_batchers=config['train']['num_batchers'])
        epoch += 1

        for batch in train_set_iter:
            train_iter += 1
            optimizer.zero_grad()

            # t1 = time.time()
            nn_util.to(batch.tensor_dict, model.device)
            # print(f'[Learner] {time.time() - t1}s took for moving tensors to device', file=sys.stderr)

            # t1 = time.time()
            result = model(batch.tensor_dict, batch.tensor_dict['prediction_target'])
            # print(f'[Learner] batch {train_iter}, {batch.size} examples took {time.time() - t1:4f}s', file=sys.stderr)

            loss = -result['batch_log_prob'].mean()

            cum_loss += loss.item() * batch.size
            cum_examples += batch.size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.)

            optimizer.step()
            del loss

            if train_iter % log_every == 0:
                log(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                    f'{cum_examples} examples ({cum_examples / (time.time() - t_log)} examples/s)')

                cum_loss = cum_examples = 0.
                t_log = time.time()

            if evaluate_every_niter > 0 and train_iter % evaluate_every_niter == 0:
                validate()

        log(f'[Learner] Epoch {epoch} finished')

        if evaluate_every_nepoch > 0 and epoch % evaluate_every_nepoch == 0:
            validate()

        if epoch == max_epoch:
            log(f'[Learner] Reached max epoch')
            exit()


def test(args):
    sys.setrecursionlimit(7000)
    is_ensemble = args['--ensemble']
    model_path = args['MODEL_FILE']
    test_set_path = args['TEST_DATA_FILE']

    extra_config = None
    if args['--extra-config']:
        extra_config = args['--extra-config']
        extra_config = json.loads(extra_config)

    print(f'loading model from [{model_path}]', file=sys.stderr)
    model_cls = EnsembleModel if is_ensemble else RenamingModel
    if is_ensemble:
        model_path = model_path.split(',')
    model = model_cls.load(model_path, use_cuda=args['--cuda'], new_config=extra_config)
    model.eval()

    test_set = Dataset(test_set_path)
    eval_results, decode_results = Evaluator.decode_and_evaluate(model, test_set, model.config, return_results=True)

    print(eval_results, file=sys.stderr)

    save_to = (args['--save-to'] if args['--save-to']
               else args['MODEL_FILE'] + f'.{test_set_path.split("/")[-1]}.decode_results.bin')
    print(f'Save decode results to {save_to}', file=sys.stderr)
    pickle.dump(decode_results, open(save_to, 'wb'))


if __name__ == '__main__':
    cmd_args = docopt(__doc__)
    print(f'Main process id {os.getpid()}', file=sys.stderr)

    # seed the RNG
    seed = int(cmd_args['--seed'])
    print(f'use random seed {seed}', file=sys.stderr)
    torch.manual_seed(seed)

    use_cuda = cmd_args['--cuda']
    if use_cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    random.seed(seed * 17 // 7)

    if cmd_args['train']:
        train(cmd_args)
    elif cmd_args['test']:
        test(cmd_args)
