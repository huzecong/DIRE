import os
import pickle
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, TypeVar

import flutes
import numpy as np
import ujson as json
from argtyped import Arguments, Switch
from tqdm import tqdm

from utils.code_processing import Lexer, tokenize_raw_code
from utils.data_compress import Example, TOKEN_DELIMITER, compress_ast


class Args(Arguments):
    input_dir: str = "../../github/decompile_output_fixed"
    file_list_cache_path: str = "file_list.pkl"
    output_dir: str = "dire_data"
    shard_size: int = 10000
    n_procs: int = 0
    pdb: Switch = False


JSON = Dict[str, Any]
VAR_ID_REGEX = re.compile(r"@@(VAR_\d+)@@")


def preprocess_ast(root: JSON, code: str) -> Dict[str, str]:
    first_line = code[:code.index('\n')]
    arg_var_ids = set(VAR_ID_REGEX.findall(first_line))
    var_map = {}

    def _visit(node: JSON) -> None:
        # Annotate type
        if node['node_type'] == 'obj' and node['type'] == 'char *':
            node['name'] = 'STRING'
        elif node['node_type'] == 'num':
            node['name'] = 'NUMBER'
        elif node['node_type'] == 'fnum':
            node['name'] = 'FLOAT'

        # Canonicalize constant
        if 'type' in node:
            type_tokens = [t[1].lstrip('_') for t in Lexer(node['type']).get_tokens()]
            type_tokens = [t for t in type_tokens if t not in ('(', ')')]
            del node['type']
            node['type_tokens'] = type_tokens

        # Annotate args
        if node['node_type'] == 'var':
            var_map[node['old_name']] = node['new_name']
            node['is_arg'] = node['var_id'] in arg_var_ids

        for key in ('x', 'y', 'z'):
            if key in node:
                _visit(node[key])
        if 'children' in node:
            for child in node['children']:
                _visit(child)

    _visit(root)
    return var_map


class ParseState(flutes.PoolState):
    def __init__(self):
        sys.setrecursionlimit(32768)
        self.duplicate = 0
        self.code_set: Set[str] = set()

    def __return_state__(self) -> int:
        return self.duplicate

    def parse_json(self, path: str) -> Iterator[Example]:
        file_name = path.split("/")[-1]
        with open(path) as f:
            for line_num, line in enumerate(f):
                if not line: continue
                try:
                    json_dict = json.loads(line)
                    raw_code = json_dict['raw_code']
                    code_tokens = tokenize_raw_code(raw_code)
                    tokenized_code = TOKEN_DELIMITER.join(code_tokens)
                    # Check if example is duplicate.
                    if tokenized_code in self.code_set:
                        self.duplicate += 1
                        continue
                    self.code_set.add(tokenized_code)

                    ast = json_dict['ast']
                    var_map = preprocess_ast(ast, code=raw_code)
                    # Check if example is valid: at least one pair of renamed variables.
                    if not any(k != v for k, v in var_map.items()):
                        continue
                    compressed_ast = compress_ast(ast)
                    example = Example(
                        function=json_dict['function'], code=tokenized_code,
                        # Pre-encode the AST so that the main process doesn't have to construct the AST before checking
                        # for duplicates.
                        ast=pickle.dumps(compressed_ast, protocol=pickle.HIGHEST_PROTOCOL),
                        file_name=file_name, line_num=line_num)
                    yield example
                except ValueError as e:
                    print(f"Exception occurred when processing line {line_num} of file {path}:")
                    print(f"<{e.__class__.__name__}> {e}")
                    traceback.print_exc()


def scandir(path: str, ext: Optional[str] = None) -> Iterator[str]:
    with os.scandir(path) as it:
        for entry in it:
            if ext is not None and not entry.name.endswith(ext):
                continue
            yield entry.path


T = TypeVar('T')


def progressed_chunk(n: int, iterable: Iterable[T]) -> Iterator[List[T]]:
    group = []
    count = 0
    progress = tqdm(total=n, desc="Shard 0")
    from flutes.log import _get_console_logging_function
    prev_log_func = _get_console_logging_function()
    flutes.set_console_logging_function(progress.write)
    for x in iterable:
        group.append(x)
        progress.update(1)
        if len(group) == n:
            yield group
            count += 1
            progress.reset()
            progress.set_description(f"Shard {count}")
            group = []
    if len(group) > 0:
        yield group
    flutes.set_console_logging_function(prev_log_func)
    progress.close()


def main():
    sys.setrecursionlimit(32768)
    np.random.seed(flutes.__MAGIC__)
    random.seed(flutes.__MAGIC__)
    args = Args()
    if args.pdb:
        flutes.register_ipython_excepthook()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_duplicates = 0
    code_set: Set[str] = set()

    def filter_fn(ex: Example) -> bool:
        if ex.code in code_set:
            nonlocal n_duplicates
            n_duplicates += 1
            return False
        code_set.add(ex.code)
        return True

    def map_fn(ex: Example) -> Example:
        return ex._replace(ast=pickle.loads(ex.ast))  # decode the AST now that we know it's not a duplicate)

    @flutes.cache(args.file_list_cache_path, verbose=True)
    def list_all_files() -> List[str]:
        return list(tqdm(scandir(args.input_dir), desc="Listing files"))

    all_files = list_all_files()
    flutes.log("Data generation begin")
    n_valid_examples = 0
    with flutes.safe_pool(args.n_procs, state_class=ParseState) as pool:
        examples_gen = map(map_fn, filter(filter_fn, pool.gather(ParseState.parse_json, all_files)))
        for idx, examples in enumerate(progressed_chunk(args.shard_size, examples_gen)):
            n_valid_examples += len(examples)
            path = output_dir / f"data_{idx:03d}.pkl"
            with path.open("wb") as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
            dup_count = n_duplicates
            total_count = dup_count + n_valid_examples
            flutes.log(f"Saved to {path}. Local duplicates: {dup_count} / {total_count} = "
                       f"{dup_count / total_count * 100:.2f}%")

        dup_count = n_duplicates + sum(pool.get_states())
        total_count = dup_count + n_valid_examples
        flutes.log(f"Global duplicates: {dup_count} / {total_count} = {dup_count / total_count * 100:.2f}%")


if __name__ == '__main__':
    main()
