import functools
import glob
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Any, NamedTuple, Optional, List, Union, Iterator

import torch
from torch.utils.data.dataloader import get_worker_info, DataLoader
from torch.utils.data.dataset import IterableDataset
from tqdm import tqdm

from utils.dataset import Dataset, Batcher, Batch, Example as ProcessedExample

__all__ = [
    "TOKEN_DELIMITER",
    "compress_ast",
    "decompress_ast",
    "Node",
    "Example",
    "CompressedDataset",
]

TOKEN_DELIMITER = "\0"
JSON = Dict[str, Any]


class Stmt(NamedTuple):
    node_id: int
    node_type: str
    address: str
    children: Optional[List['Node']] = None
    parent_address: Optional[str] = None


class Literal(NamedTuple):
    node_id: int
    node_type: str
    type_tokens: str
    address: str
    name: Optional[str] = None  # for some reason, certain "fnum" nodes have missing values
    ref_width: Optional[int] = None
    parent_address: Optional[str] = None


class Expr(NamedTuple):
    node_id: int
    node_type: str
    address: str
    type_tokens: str
    x: Optional['Node'] = None  # somehow "sizeof" has only "children" attr
    # y: Optional['Node'] = None
    # z: Optional['Node'] = None
    # children: Optional[List['Node']] = None
    parent_address: Optional[str] = None
    extra_attrs: Optional[Dict[str, Any]] = None


class Var(NamedTuple):
    node_id: int
    node_type: str
    var_id: str
    old_name: str
    new_name: str
    address: str
    type_tokens: str
    ref_width: int
    is_arg: bool
    parent_address: Optional[str] = None


Node = Union[Stmt, Literal, Expr, Var]


class Example(NamedTuple):
    function: str  # func_name
    code: str  # tokenized code, concat'ed by "\0"
    ast: Node
    file_name: str
    line_num: int


EXPR_FIELDS = set(Expr._fields)


def compress_ast(ast: JSON) -> Node:
    args = ast.copy()
    if "type_tokens" in args:
        args["type_tokens"] = TOKEN_DELIMITER.join(args["type_tokens"])
    for key, value in args.items():
        if isinstance(value, dict):
            args[key] = compress_ast(value)
        elif isinstance(value, list):
            args[key] = [compress_ast(x) for x in value]

    if "var_id" in ast:
        type_class = Var
    elif "x" in ast or ast["node_type"] in {"sizeof"}:
        new_args = {}
        for key, value in args.items():
            if key in EXPR_FIELDS:
                new_args[key] = value
            else:
                new_args.setdefault("extra_attrs", {})[key] = value
        args = new_args
        type_class = Expr
    elif "type_tokens" in ast:
        type_class = Literal
    else:
        type_class = Stmt
    return type_class(**args)


def decompress_ast(ast: Node) -> JSON:
    j = {}
    for key, value in zip(ast._fields, ast):
        if value is not None:
            j[key] = value
    if "type_tokens" in j:
        j["type_tokens"] = j["type_tokens"].split(TOKEN_DELIMITER)
    if isinstance(ast, Expr) and ast.extra_attrs is not None:
        j.update(ast.extra_attrs)
        del j["extra_attrs"]
    for key, value in j.items():
        if isinstance(value, tuple):
            j[key] = decompress_ast(value)
    if "children" in j:
        j["children"] = [decompress_ast(x) for x in j["children"]]
    return j


Config = Dict[str, Any]


class IterDataset(IterableDataset):
    batcher: Batcher

    def __init__(self, file_paths: List[Path], mode: str, collate: bool, shuffle: bool = True,
                 batcher_config: Optional[Config] = None, max_tokens_per_batch: Optional[int] = None,
                 return_examples: bool = False, return_prediction_target: bool = True):
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.mode = mode
        self.batcher_config = batcher_config
        self.collate = collate
        self.max_tokens_per_batch = max_tokens_per_batch
        self.return_examples = return_examples
        self.return_prediction_target = return_prediction_target

    def process(self, raw_ex: Example) -> Optional[ProcessedExample]:
        json_dict = {
            "function": raw_ex.function,
            "ast": decompress_ast(raw_ex.ast),
        }
        meta = {
            "file_name": raw_ex.file_name,
            "line_num": raw_ex.line_num,
        }
        ex = ProcessedExample.from_json_dict(
            json_dict, binary_file=meta, code_tokens=raw_ex.code.split(TOKEN_DELIMITER))
        if self.collate:
            self.batcher.annotate_example(ex)
            if self.mode == "train" and ex.target_prediction_seq_length >= 200:
                return None
        if self.mode == "train" and ex.ast.size >= 300:
            return None
        return ex

    def _generate_batches(self, buffer: List[ProcessedExample], shuffle: bool) -> Iterator[Batch]:
        self.batcher.sort_training_examples(buffer)
        batches = []
        batch_examples = []
        for example in buffer:
            batch_size_with_example = self.batcher.get_batch_size(batch_examples + [example])
            if batch_examples and batch_size_with_example > self.max_tokens_per_batch:
                batches.append(batch_examples)
                batch_examples = []
            batch_examples.append(example)
        if len(batch_examples) > 0:
            batches.append(batch_examples)
        if shuffle:
            random.shuffle(batches)

        for examples in batches:
            batch = self.batcher.to_batch(
                examples, return_examples=self.return_examples,
                return_prediction_target=self.return_prediction_target)
            yield batch

    def iterate_dataset(self, shuffle: Optional[bool] = None) -> Union[Iterator[ProcessedExample], Iterator[Batch]]:
        # Resource initialization is postponed to this point, so that the resources are initialized on the worker
        # processes.
        if shuffle is None:
            shuffle = self.shuffle

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            split_size = len(self.file_paths) // worker_info.num_workers
            files = self.file_paths[(split_size * worker_id):(split_size * (worker_id + 1))]
        else:
            worker_id = "main"
            files = self.file_paths.copy()

        if shuffle:
            random.shuffle(files)
        if self.collate:
            self.batcher = Batcher(self.batcher_config, train=(self.mode == "train"))
        buffer = []
        buffer_size = self.batcher_config['train']['buffer_size']
        for file in files:
            with file.open("rb") as f:
                data: List[Example] = pickle.load(f)
            # print(f"Worker {worker_id}: Loaded file {file}", flush=True)
            if shuffle:
                random.shuffle(data)
            for raw_ex in data:
                ex = self.process(raw_ex)
                if ex is None: continue

                if self.collate:
                    buffer.append(ex)
                    if len(buffer) >= buffer_size:
                        yield from self._generate_batches(buffer, shuffle)
                        buffer = []
                else:
                    yield ex
        if self.collate and len(buffer) > 0:
            yield from self._generate_batches(buffer, shuffle)

    def __iter__(self) -> Iterator[ProcessedExample]:
        return self.iterate_dataset(shuffle=self.shuffle)


def worker_init_fn(worker_id: int, random_seed: Optional[int]) -> None:
    if random_seed is not None:
        import numpy as np
        seed = random_seed + worker_id
        random.seed(seed * 17 // 7)
        np.random.seed(seed * 13 // 7)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


class CompressedDataset(Dataset):
    def __init__(self, file_paths: List[str], random_seed: Optional[int] = None) -> None:
        if not isinstance(file_paths, list):
            assert isinstance(file_paths, str)
            file_paths = glob.glob(file_paths)
        self.file_paths = [Path(path) for path in file_paths]
        self.random_seed = random_seed
        self.size = len(self.file_paths)  # a fake size

    def _create_data_iter(self, shuffle: bool, collate: bool, num_workers: int = 1, progress: bool = True, **kwargs):
        dataset = IterDataset(self.file_paths, shuffle=shuffle, collate=collate, **kwargs)
        dataloader = DataLoader(
            dataset, batch_size=None, num_workers=num_workers, pin_memory=True,
            worker_init_fn=functools.partial(worker_init_fn, random_seed=self.random_seed))
        # if progress:
        #     dataloader = tqdm(dataloader, file=sys.stdout)
        return dataloader

    def get_iterator(self, shuffle=False, progress=True, num_workers=1) -> Iterator[ProcessedExample]:
        return self._create_data_iter(
            shuffle=shuffle, collate=False, mode="eval",
            num_workers=num_workers, progress=progress)

    def batch_iterator(self,  # type: ignore
                       batch_size: int, config: Config, return_examples=False, return_prediction_target=None,
                       num_readers=3, num_batchers=3, progress=True, train=False, **kwargs) \
            -> Iterator[Batch]:
        return self._create_data_iter(
            shuffle=train, collate=True, mode=("train" if train else "eval"),
            num_workers=num_batchers, progress=progress,
            batcher_config=config, max_tokens_per_batch=batch_size,
            return_examples=return_examples, return_prediction_target=return_prediction_target)
