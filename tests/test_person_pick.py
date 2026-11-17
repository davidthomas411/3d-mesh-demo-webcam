"""When several people are in view the server follows one: the person it was already on, else the largest central one."""
import ast, os
import numpy as np

SRC = open(os.path.join(os.path.dirname(__file__), "..", "webcam_server.py"), encoding="utf-8").read()


def load():
    tree = ast.parse(SRC)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("box_iou", "pick_person")]
    ns = {"np": np}
    exec(compile(ast.Module(keep, []), "webcam_server_excerpt", "exec"), ns)
    return ns["pick_person"]


def test_largest_central_person_wins_and_the_followed_person_is_kept():
    pick = load()
    shape = (480, 640, 3)
    centre, edge_big, small = [220, 80, 420, 460], [0, 40, 230, 470], [500, 200, 560, 330]
    assert pick([small, centre], shape) == 1
    assert pick([edge_big, centre], shape) == 1                 # slightly larger, but at the edge
    assert pick([centre, small], shape, last=[505, 205, 565, 335]) == 1      # stays with the person already followed
    assert pick([centre, small], shape, last=[0, 0, 10, 10]) == 0            # that person has gone: start again
    assert pick([centre], shape) == 0
