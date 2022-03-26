#!/usr/bin/env python

"""Tests for `nabla` package."""
from nabla.graph import Graph, Variable, multiply, Constant, add
from nabla.compute import topological_sort, forward_pass, backward_pass


def test_topological_sort():
    with Graph():
        x = Variable(1.3, name="x")
        y = Variable(1, name="y")
        z = Constant(5, name="z")

        f = x * y + z

        order = topological_sort(f)
        assert len(order) == 5

        expected_types = [Variable, Variable, multiply, Constant, add]
        actual_types = [type(x) for x in order]
        assert actual_types == expected_types

        expected_names = ["x", "y", "mul/0", "z", "add/0"]
        actual_names = [x.name for x in order]
        assert actual_names == expected_names


def test_forward_pass():
    with Graph():
        x = Variable(1.3, name="x")
        y = Variable(1, name="y")
        z = 5

        f = x * y + z

        order = topological_sort(f)
        result = forward_pass(order)
        assert result == 6.3


def test_backward_pass():
    with Graph():
        x = Variable(1.3, name="x")
        y = Variable(1, name="y")
        z = 5

        f = x * y + z

        order = topological_sort(f)
        _ = forward_pass(order)
        _ = backward_pass(order)
