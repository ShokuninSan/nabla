# -*- coding: utf-8 -*-
from .graph import Operator, Placeholder


def topological_sort(head_node):
    vis = set()
    ordering = []

    def _dfs(node):
        if node not in vis:
            vis.add(node)
            if isinstance(node, Operator):
                for input_node in node.inputs:
                    _dfs(input_node)
            ordering.append(node)

    _dfs(head_node)

    return ordering


def forward_pass(order, feed_dict={}):
    for node in order:

        if isinstance(node, Placeholder):
            node.value = feed_dict[node.name]

        elif isinstance(node, Operator):
            node.value = node.forward(
                *[prev_node.value for prev_node in node.inputs]
            )

    return order[-1].value


def backward_pass(order, target_node=None):
    vis = set()
    order[-1].gradient = 1
    for node in reversed(order):
        if isinstance(node, Operator):
            inputs = node.inputs
            grads = node.backward(
                *[x.value for x in inputs], dout=node.gradient
            )
            for inp, grad in zip(inputs, grads):
                if inp not in vis:
                    inp.gradient = grad
                else:
                    inp.gradient += grad
                vis.add(inp)
    return [node.gradient for node in order]
