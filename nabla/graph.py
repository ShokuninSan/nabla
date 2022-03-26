"""Main module."""
from abc import ABC

import numpy as np


class Graph:
    def __init__(self):
        self.operators = set()
        self.constants = set()
        self.variables = set()
        self.placeholders = set()
        global _g
        _g = self

    def reset_counts(self, root):
        if hasattr(root, "count"):
            root.count = 0
        else:
            for child in root.__subclasses__():
                self.reset_counts(child)

    def reset_session(self):
        try:
            del _g  # noqa: F821
        except NameError:
            pass
        self.reset_counts(Node)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.reset_session()


class Node(ABC):
    pass


class Placeholder(Node):
    count = 0

    def __init__(self, name, dtype=float):
        _g.placeholders.add(self)
        self.value = None
        self.gradient = None
        self.name = f"Plc/{Placeholder.count}" if name is None else name
        Placeholder.count += 1

    def __repr__(self):
        return f"Placeholder: name:{self.name}, value:{self.value}"


class Constant(Node):
    count = 0

    def __init__(self, value, name=None):
        _g.constants.add(self)
        self._value = value
        self.gradient = None
        self.name = f"Const/{Constant.count}" if name is None else name
        Constant.count += 1

    def __repr__(self):
        return f"Constant: name:{self.name}, value:{self.value}"

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self):
        raise ValueError("Cannot reassign constant")


class Variable(Node):
    count = 0

    def __init__(self, value, name=None):
        _g.variables.add(self)
        self.value = value
        self.gradient = None
        self.name = f"Var/{Variable.count}" if name is None else name
        Variable.count += 1

    def __repr__(self):
        return f"Variable: name:{self.name}, value:{self.value}"


class Operator(Node):
    def __init__(self, name="Operator"):
        _g.operators.add(self)
        self.value = None
        self.inputs = []
        self.gradient = None
        self.name = name

    def __repr__(self):
        return f"Operator: name:{self.name}"


class add(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"add/{add.count}" if name is None else name
        add.count += 1

    def forward(self, a, b):
        return a + b

    def backward(self, a, b, dout):
        return dout, dout


class multiply(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"mul/{multiply.count}" if name is None else name
        multiply.count += 1

    def forward(self, a, b):
        return a * b

    def backward(self, a, b, dout):
        return dout * b, dout * a


class divide(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"div/{divide.count}" if name is None else name
        divide.count += 1

    def forward(self, a, b):
        return a / b

    def backward(self, a, b, dout):
        return dout / b, dout * a / np.power(b, 2)


class power(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"pow/{power.count}" if name is None else name
        power.count += 1

    def forward(self, a, b):
        return np.power(a, b)

    def backward(self, a, b, dout):
        return (
            dout * b * np.power(a, (b - 1)),
            dout * np.log(a) * np.power(a, b),
        )


class matmul(Operator):
    count = 0

    def __init__(self, a, b, name=None):
        super().__init__(name)
        self.inputs = [a, b]
        self.name = f"matmul/{matmul.count}" if name is None else name
        matmul.count += 1

    def forward(self, a, b):
        return a @ b

    def backward(self, a, b, dout):
        return dout @ b.T, a.T @ dout


def node_wrapper(func, self, other):
    if isinstance(other, Node):
        return func(self, other)
    if isinstance(other, float) or isinstance(other, int):
        return func(self, Constant(other))
    raise TypeError("Incompatible types.")


Node.__add__ = lambda self, other: node_wrapper(add, self, other)
Node.__mul__ = lambda self, other: node_wrapper(multiply, self, other)
Node.__div__ = lambda self, other: node_wrapper(divide, self, other)
Node.__neg__ = lambda self: node_wrapper(multiply, self, Constant(-1))
Node.__pow__ = lambda self, other: node_wrapper(power, self, other)
Node.__matmul__ = lambda self, other: node_wrapper(matmul, self, other)
