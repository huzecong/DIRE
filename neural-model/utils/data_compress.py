from typing import Dict, Any, NamedTuple, Optional, List, Union

__all__ = [
    "TOKEN_DELIMITER",
    "compress_ast",
    "decompress_ast",
    "Node",
    "Example",
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
