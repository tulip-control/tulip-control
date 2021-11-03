"""Input from TLA+."""
import collections

from tla import parser
from tla import to_str
from tla import visit


COLUMN_WIDTH = 80


def parse(module_source):
    """Return namespace of definitions in `module`.

    Parses the TLA+ string `module` and returns
    a dictionary that maps each defined operator
    to a dictionary `dict(parameters=[...], definiens=...)`
    of the list of parameter names (each a `str`),
    and of the definiens expression (as `str`).

    In each definiens, occurrences of already defined
    operators have been recursively expanded.
    Second-order operators and `LET` expressions
    are currently not expanded.

    Example:

    ```python
    import tulip.spec._tla as _tla


    tla_source = '''
    ---- MODULE Example ----

    VARIABLE z

    A == TRUE
    B(x) == x
    C(y) == A => B(y)

    P == C(z)

    ========================
    '''

    namespace = _tla.parse(tla_source)
    print(namespace)
    ```

    prints (formatted by editing the output):

    ```
    {
        'A': {'parameters': [],
              'definiens': 'TRUE'},
        'B': {'parameters': ['x'],
              'definiens': 'x'},
        'C': {'parameters': ['y'],
              'definiens': 'A => B(y)'},
        'P': {'parameters': [],
              'definiens': 'C(z)'}
    }
    ```

    @param module: string that describes
        a TLA+ module
    @type module: `str`
    @return: `dict` that maps each defined
        operator to its signature and definiens
    @rtype: `dict(str -> op_def)`
        where `op_def` is
        `dict(parameters=names, definiens=expr)`
        with:
        - `names` a `list` of `str`
        - `expr` a `str`
    """
    tree = parser.parse(
        module_source, nodes=to_str.Nodes)
    # expand defined operators
    # and store in namespace
    namespace = dict()
    expander = _ExpandDefinedOperators()
    expander.visit(tree, namespace=namespace)
    to_infix = _ConvertToInfix()
    new_namespace = dict()
    for opname, op_def in namespace.items():
        print(op_def.definiens)
        definiens_tree = to_infix.visit(
            op_def.definiens)
        definiens = definiens_tree.to_str(
            width=COLUMN_WIDTH)
        new_namespace[opname] = dict(
            parameters=op_def.parameters,
            definiens=definiens)
    return new_namespace


class _ExpandDefinedOperators(visit.NodeTransformer):
    """Replace each defined operator by its definiens."""

    def visit_Opaque(self, node, *arg, **kw):
        """Operator name."""
        namespace = kw['namespace']
        name = node.name
        if name not in namespace:
            return self.nodes.Opaque(name)
        op_def = namespace[name]
        if op_def.parameters:
            return self.nodes.Opaque(name)
        else:
            return _parenthesize(op_def.definiens, self.nodes)

    def visit_Apply(self, node, *arg, **kw):
        """Operator application."""
        op = self.visit(node.op, *arg, **kw)
        args = [
            self.visit(u, *arg, **kw)
            for u in node.operands]
        namespace = kw['namespace']
        is_defined_op = (
            type(op).__name__ == 'Opaque' and
            op.name in namespace)
        if not is_defined_op:
            return self.nodes.Apply(op, args)
        op_def = namespace[op.name]
        if not op_def.parameters:
            raise AssertionError(
                f'operator `{op.name}` is applied '
                'to arguments, but appears to '
                'be nullary')
        sub = dict()
        for param, expr in zip(op_def.parameters, args):
            sub[param] = _OperatorDef(
                parameters=list(),
                definiens=expr)
        new_namespace = _extend_namespace(sub, namespace)
        new_kw = dict(kw)
        new_kw['namespace'] = new_namespace
        # beta-reduction
        reduced = self.visit(
            op_def.definiens, *arg, **new_kw)
        return reduced

    def visit_Definition(self, node, *arg, **kw):
        """Module-scope definition."""
        namespace = kw['namespace']
        defn = node.definition
        name = defn.name
        expr = self.visit(defn.expr, *arg, **kw)
        is_lambda = type(expr).__name__ == 'Lambda'
        if is_lambda:
            parameters = [
                name for name, _ in expr.name_shapes]
            op_def = _OperatorDef(
                parameters=parameters,
                definiens=expr.expr)
        else:
            parameters = list()
            op_def = _OperatorDef(
                parameters=parameters,
                definiens=expr)
        op_defs = {name: op_def}
        _extend_namespace(
            op_defs, namespace, inplace=True)
        return expr


def _extend_namespace(op_defs, namespace, inplace=False):
    """Return new namespace that extends `namespace`.

    If `inplace is True`, then mutate `namespace`,
    otherwise create a shallow of copy of `namespace`,
    and mutate that.
    """
    if inplace:
        out_namespace = namespace
    else:
        out_namespace = dict(namespace)
    for name, op_def in op_defs.items():
        if name in out_namespace:
            raise AssertionError(
                f'operator `{name}` redefined '
                f'in: {out_namespace}')
        out_namespace[name] = op_def
    return out_namespace


class _ConvertToInfix(visit.NodeTransformer):
    """Convert junctions to infix conj/disjunction."""

    def visit_List(self, node, *arg, **kw):
        """Conjunction or disjunction list."""
        op = self.visit(node.op, *arg, **kw)
        exprs = [
            self.visit(e, *arg, **kw)
            for e in node.exprs]
        exprs = [
            _parenthesize(e, self.nodes)
            for e in exprs]
        # convert to infix conjunctions or disjunctions
        opname = type(op).__name__
        if opname == 'And':
            op_node = self.nodes.Conj()
        elif opname == 'Or':
            op_node = self.nodes.Disj()
        else:
            raise ValueError(op)
        infix_expr = exprs[0]
        for item in exprs[1:]:
            infix_op = self.nodes.Internal(op_node)
            args = [infix_expr, item]
            infix_expr = self.nodes.Apply(infix_op, args)
        return infix_expr


def _parenthesize(expr, nodes):
    """Wrap `expr` inside parentheses.

    @type expr: syntax tree
    @rtype: syntax tree
    """
    return nodes.Parens(expr, nodes.Syntax())


_OperatorDef = collections.namedtuple(
    '_OperatorDef', 'parameters, definiens')
