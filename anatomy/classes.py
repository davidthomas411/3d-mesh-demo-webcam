"""TotalSegmentator v2 'total' task: grouping + colours for the 117 classes."""
import re

GROUPS = [
    ("skull",        r"^skull$"),
    ("spine",        r"^(vertebrae_|sacrum)"),
    ("ribcage",      r"^(rib_|sternum|costal_cartilages)"),
    ("shoulder",     r"^(clavicula|scapula|humerus)"),
    ("pelvis_leg",   r"^(hip_|femur)"),
    ("muscle",       r"^(gluteus|iliopsoas|autochthon)"),
    ("heart",        r"^(heart|atrial_appendage)"),
    ("artery",       r"(aorta|artery|brachiocephalic_trunk)"),
    ("vein",         r"(vein|vena_cava)"),
    ("lung",         r"^lung_"),
    ("airway",       r"^trachea"),
    ("gi",           r"^(esophagus|stomach|duodenum|small_bowel|colon|liver|gallbladder|pancreas|spleen)"),
    ("urinary",      r"^(kidney|urinary_bladder|adrenal|prostate)"),
    ("neuro_endo",   r"^(brain|spinal_cord|thyroid|face)"),
]
GROUP_COLOR = {
    "skull": (0.93, 0.91, 0.84), "spine": (0.95, 0.93, 0.80), "ribcage": (0.92, 0.90, 0.78),
    "shoulder": (0.90, 0.88, 0.76), "pelvis_leg": (0.88, 0.86, 0.74), "muscle": (0.72, 0.25, 0.22),
    "heart": (0.80, 0.15, 0.20), "artery": (0.90, 0.20, 0.20), "vein": (0.25, 0.35, 0.80),
    "lung": (0.95, 0.65, 0.70), "airway": (0.55, 0.80, 0.90), "gi": (0.78, 0.55, 0.30),
    "urinary": (0.85, 0.75, 0.30), "neuro_endo": (0.70, 0.60, 0.85), "other": (0.6, 0.6, 0.6),
}
# Expected localisation error vs a REAL patient, from atlas-to-skin registration (illustrative priors,
# to be replaced by leave-one-out numbers, see README): bone near the skin vs deep soft tissue.
UNCERTAINTY_MM = {"skull": 10, "spine": 15, "ribcage": 15, "shoulder": 20, "pelvis_leg": 15, "muscle": 15,
                  "heart": 30, "artery": 30, "vein": 30, "lung": 25, "airway": 20, "gi": 35,
                  "urinary": 35, "neuro_endo": 20, "other": 40}


def group_of(name):
    for g, pat in GROUPS:
        if re.search(pat, name):
            return g
    return "other"


def display_name(name):
    return name.replace("_", " ")
