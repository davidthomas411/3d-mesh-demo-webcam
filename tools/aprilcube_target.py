"""The calibration cube: an aprilcube target (https://github.com/younghyopark/aprilcube, MIT).

aprilcube prints a cuboid with a grid of ArUco / AprilTag markers on every face and writes a config.json describing it.
This module reads that config (or builds one from the same parameters), lays the markers' corners out in the cube's own
frame exactly as aprilcube does, finds the markers in a picture with OpenCV and solves the cube's pose from all of them
at once. Nothing is installed: the layout rules are short and are restated here.

Cube frame: origin at the cube's centre, millimetres, axes along the +X, +Y, +Z faces' outward normals.
Marker ids run face by face in the order +X, -X, +Y, -Y, +Z, -Z, row by row within a face.
"""
import json
import numpy as np
import cv2

# name, normal axis, sign, right axis, sign, down axis, sign  (aprilcube generate.py FACE_DEFS)
FACE_DEFS = [("+X", 0, +1, 1, -1, 2, -1), ("-X", 0, -1, 1, +1, 2, -1), ("+Y", 1, +1, 0, +1, 2, -1),
             ("-Y", 1, -1, 0, -1, 2, -1), ("+Z", 2, +1, 0, +1, 1, +1), ("-Z", 2, -1, 0, +1, 1, -1)]
DICTS = {"4x4_50": cv2.aruco.DICT_4X4_50, "4x4_100": cv2.aruco.DICT_4X4_100, "4x4_250": cv2.aruco.DICT_4X4_250,
         "4x4_1000": cv2.aruco.DICT_4X4_1000, "5x5_50": cv2.aruco.DICT_5X5_50, "5x5_100": cv2.aruco.DICT_5X5_100,
         "5x5_250": cv2.aruco.DICT_5X5_250, "5x5_1000": cv2.aruco.DICT_5X5_1000, "6x6_50": cv2.aruco.DICT_6X6_50,
         "6x6_100": cv2.aruco.DICT_6X6_100, "6x6_250": cv2.aruco.DICT_6X6_250, "6x6_1000": cv2.aruco.DICT_6X6_1000,
         "7x7_50": cv2.aruco.DICT_7X7_50, "7x7_100": cv2.aruco.DICT_7X7_100, "7x7_250": cv2.aruco.DICT_7X7_250,
         "7x7_1000": cv2.aruco.DICT_7X7_1000, "aruco_original": cv2.aruco.DICT_ARUCO_ORIGINAL,
         "apriltag_16h5": cv2.aruco.DICT_APRILTAG_16h5, "apriltag_25h9": cv2.aruco.DICT_APRILTAG_25h9,
         "apriltag_36h10": cv2.aruco.DICT_APRILTAG_36h10, "apriltag_36h11": cv2.aruco.DICT_APRILTAG_36h11}


class Cube:
    def __init__(self, dict_name="6x6_100", grid=(3, 3, 3), tag_size_mm=30.0, margin_cells=1.0, border_cells=1.0,
                 first_id=0, tag_ids=None):
        self.dict_name, self.grid = dict_name, tuple(int(g) for g in grid)
        self.dictionary = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])
        self.mp = self.dictionary.markerSize + 2                 # cells across one marker, with its black border
        self.tag, self.margin, self.border = float(tag_size_mm), float(margin_cells), float(border_cells)
        self.cs = self.tag / self.mp
        self.cells = [2 * self.border + g * self.mp + (g - 1) * self.margin for g in self.grid]
        self.dims = np.array(self.cells) * self.cs               # the box, mm
        n = 2 * (self.grid[0] * self.grid[1] + self.grid[0] * self.grid[2] + self.grid[1] * self.grid[2])
        self.ids = list(tag_ids) if tag_ids is not None else list(range(first_id, first_id + n))
        self.corners, self.face_of = self._layout()

    @classmethod
    def from_config(cls, path):
        """An aprilcube config.json."""
        d = json.load(open(path))
        return cls(d["dict"], [int(x) for x in str(d["grid"]).lower().split("x")] if isinstance(d["grid"], str) else d["grid"],
                   d["tag_size_mm"], d.get("margin_cells", 1), d.get("border_cells", 1), tag_ids=d.get("tag_ids"))

    def config(self):
        return dict(dict=self.dict_name, grid="%dx%dx%d" % self.grid, tag_ids=self.ids, tag_size_mm=self.tag,
                    margin_cells=self.margin, border_cells=self.border, cell_size_mm=self.cs,
                    box_dims=[float(x) for x in self.dims], marker_pixels=self.mp)

    def _layout(self):
        corners, face_of, k = {}, {}, 0
        for name, na, ns, ra, rs_, da, ds in FACE_DEFS:
            cols, rows = self.grid[ra], self.grid[da]
            rh, dh = self.dims[ra] / 2, self.dims[da] / 2

            def at(row, col):
                p = np.zeros(3); p[na] = ns * self.dims[na] / 2
                p[ra] = rs_ * (-rh + col * self.cs); p[da] = ds * (-dh + row * self.cs)
                return p
            for r in range(rows):
                for c in range(cols):
                    r0 = self.border + r * (self.mp + self.margin); c0 = self.border + c * (self.mp + self.margin)
                    m = self.mp                                   # patterns are mirrored when printed, hence the order
                    corners[self.ids[k]] = np.array([at(r0, c0 + m), at(r0, c0), at(r0 + m, c0), at(r0 + m, c0 + m)])
                    face_of[self.ids[k]] = name; k += 1
        return corners, face_of

    def faces_mm(self):
        """Each face as (name, centre, outward normal)."""
        out = []
        for name, na, ns, *_ in FACE_DEFS:
            n = np.zeros(3); n[na] = ns; out.append((name, n * self.dims[na] / 2, n))
        return out


class CubeFinder:
    """Markers in a picture -> the cube's pose in the camera's frame (X_cam = R X_cube + t, mm)."""

    def __init__(self, cube, K, dist=None):
        self.cube, self.K = cube, np.asarray(K, np.float64)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
        p = cv2.aruco.DetectorParameters()                        # aprilcube's settings
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 3, 53, 4
        p.minMarkerPerimeterRate = 0.01; p.perspectiveRemovePixelPerCell = 8; p.errorCorrectionRate = 0.6
        self.detector = cv2.aruco.ArucoDetector(cube.dictionary, p)
        self.clahe = cv2.createCLAHE(2.0, (8, 8))

    def detect(self, bgr):
        g = self.clahe.apply(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr)
        c, ids, _ = self.detector.detectMarkers(g)
        if ids is None:
            return np.zeros(0, int), np.zeros((0, 4, 2))
        keep = [j for j, k in enumerate(ids.ravel()) if int(k) in self.cube.corners]
        return ids.ravel()[keep].astype(int), np.array([c[j].reshape(4, 2) for j in keep]).reshape(-1, 4, 2)

    def solve(self, ids, corners, guess=None, min_tags=2):
        """Pose from detected markers. Returns dict(R, t, err px, n tags used, faces) or None."""
        if len(ids) < min_tags:
            return None
        obj = np.concatenate([self.cube.corners[int(k)] for k in ids]); img = corners.reshape(-1, 2).astype(np.float64)
        try:
            if guess is not None:
                ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, self.K, self.dist, cv2.Rodrigues(guess[0])[0], guess[1].reshape(3, 1).copy(),
                                                     True, 200, 3.0, 0.99, flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                ok, rv, tv, inl = cv2.solvePnPRansac(obj, img, self.K, self.dist, iterationsCount=200, reprojectionError=3.0,
                                                     confidence=0.99, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            return None
        if not ok or inl is None or len(inl) < 8:
            return None
        inl = inl.ravel()
        rv, tv = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, self.dist, rv, tv)
        pr = cv2.projectPoints(obj[inl], rv, tv, self.K, self.dist)[0].reshape(-1, 2)
        used = sorted({int(ids[j // 4]) for j in inl})
        return dict(R=cv2.Rodrigues(rv)[0], t=tv.ravel(), err=float(np.linalg.norm(pr - img[inl], axis=1).mean()),
                    n=len(used), points=len(inl), faces=sorted({self.cube.face_of[k] for k in used}))

    def pose(self, bgr, guess=None):
        ids, corners = self.detect(bgr)
        return self.solve(ids, corners, guess)
