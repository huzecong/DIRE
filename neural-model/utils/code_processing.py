import re

from utils.ast import SyntaxNode
from utils.lexer import *

VARIABLE_ANNOTATION = re.compile(r'@@\w+@@(\w+)@@\w+')
CANONICALIZE_REGEX_1 = re.compile(r'//.*?\n|/\*.*?\*/', flags=re.S)
CANONICALIZE_REGEX_2 = re.compile(r'@@\w+@@(\w+)@@\w+')

def canonicalize_code(code):
    code = CANONICALIZE_REGEX_1.sub(r'\n', code)
    lines = [l.rstrip() for l in code.split(r'\n')]
    code = r'\n'.join(lines)
    code = CANONICALIZE_REGEX_2.sub(r'\g<1>', code)
    return code


def canonicalize_constants(root: SyntaxNode) -> None:
    def _visit(node):
        if node.node_type == 'obj' and node.type == 'char *':
            node.name = 'STRING'
        elif node.node_type == 'num':
            node.name = 'NUMBER'
        elif node.node_type == 'fnum':
            node.name = 'FLOAT'

        for child in node.member_nodes:
            _visit(child)

    _visit(root)


def annotate_type(root: SyntaxNode) -> None:
    def _visit(node):
        if hasattr(node, 'type'):
            type_tokens = [t[1].lstrip('_') for t in Lexer(node.type).get_tokens()]
            type_tokens = [t for t in type_tokens if t not in ('(', ')')]
            node.named_fields.add('type_tokens')
            setattr(node, 'type_tokens', type_tokens)

        for child in node.member_nodes:
            _visit(child)

    _visit(root)


VAR_ID_REGEX = re.compile(r"@@(VAR_\d+)@@")


def preprocess_ast(root: SyntaxNode, code: str = None,
                   annotate_type: bool = True, canonicalize_constant: bool = True, annotate_arg: bool = True) -> None:
    arg_var_ids = None
    if annotate_arg:
        first_line = code[:code.index('\n')]
        arg_var_ids = set(VAR_ID_REGEX.findall(first_line))

    def _visit(node):
        if annotate_type:
            if node.node_type == 'obj' and node.type == 'char *':
                node.name = 'STRING'
            elif node.node_type == 'num':
                node.name = 'NUMBER'
            elif node.node_type == 'fnum':
                node.name = 'FLOAT'

        if canonicalize_constant:
            if hasattr(node, 'type'):
                type_tokens = [t[1].lstrip('_') for t in Lexer(node.type).get_tokens()]
                type_tokens = [t for t in type_tokens if t not in ('(', ')')]
                node.named_fields.add('type_tokens')
                setattr(node, 'type_tokens', type_tokens)

        if annotate_arg:
            if node.node_type == 'var':
                node.named_fields.add('is_arg')
                setattr(node, 'is_arg', node.var_id in arg_var_ids)

        for child in node.member_nodes:
            _visit(child)

    _visit(root)


def tokenize_raw_code(raw_code):
    lexer = Lexer(raw_code)
    tokens = []
    for token_type, token in lexer.get_tokens():
        if token_type in Token.Literal:
            token = str(token_type).split('.')[2]

        if token_type == Token.Placeholder.Var:
            m = VARIABLE_ANNOTATION.match(token)
            old_name = m.group(1)
            token = '@@' + old_name + '@@'

        tokens.append(token)

    return tokens
