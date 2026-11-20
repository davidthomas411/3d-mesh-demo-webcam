"""Real-time layer: the display-rate pose stream (anatomy/stream.js) and the browser rig's speed on this machine."""
import os, shutil, subprocess
import pytest

HERE = os.path.dirname(__file__)
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def test_rig_stream_extrapolates_but_respects_the_horizon():
    r = subprocess.run(["node", os.path.join(HERE, "js", "stream_test.mjs")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
