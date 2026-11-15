"""SYNTHETIC test clip (not motion capture): a smooth reach-and-turn built from hand-set MHR parameters, to exercise the clip pipeline.
Replace with a real SAM 3D Body clip (tools/sam3d_video_to_clip.py) for the real thing. usage: python clips/make_synthetic_clip.py OUT.npz [frames]"""
import sys
import numpy as np

T = int(sys.argv[2]) if len(sys.argv) > 2 else 24
t = np.linspace(0, 1, T)
ease = lambda x: 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))
P = np.zeros((T, 204), np.float32)
bend = ease(np.sin(np.pi * t) ** 1.0)                       # lean forward and come back
turn = np.sin(2 * np.pi * t)                                # twist one way, then the other
for i in (10, 16, 20, 23):   P[:, i] = 0.10 * bend          # forward bend shared along the trunk
for i in (6, 12, 18, 21):    P[:, i] = 0.08 * turn          # twist shared along the trunk
for i in (24, 25):           P[:, i] = 0.25 * bend          # head/neck flexion with the lean
P[:, 31] = 0.25 * ease(np.sin(np.pi * np.clip(t * 1.4, 0, 1)))   # right shoulder girdle lifts
P[:, 30] = -0.9 * ease(np.sin(np.pi * np.clip(t * 1.4, 0, 1)))   # right arm reaches
np.savez(sys.argv[1], mhr_model_params=P, fps=12.0, source="SYNTHETIC hand-set parameters, not motion capture")
print("wrote", sys.argv[1], P.shape)
