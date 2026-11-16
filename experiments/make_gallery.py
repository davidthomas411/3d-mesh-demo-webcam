"""Build the motion gallery: for each Rocketbox clip, retarget to MHR, render the rig + CT frames, put the original motion beside them, encode a video and a
poster, and write docs/data/motions.json. Resumable (finished clips are skipped).
usage: ATLAS=.. REST=.. ONNX=.. SUBJECT=.. CORR=.. python experiments/make_gallery.py WORKDIR SITEDIR [clip ...]"""
import os, sys, json, math, subprocess, urllib.request, shutil
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
HF = "https://huggingface.co/datasets/codepawl/rocketbox-motions-glb/resolve/main"
TARGET_FRAMES = 16
CLIPS = {
    "f-gestic-shrug-01": ("Shrug", "A conversational shrug: shoulders up, head tilted, palms out."),
    "f-idle-look-around-01": ("Look around", "The head turns well past 90 degrees each way: the hardest test of the neck and skull."),
    "f-idle-stretch-arms-01": ("Stretch the arms", "Both arms reach up and back: shoulder girdle, scapulae and upper spine."),
    "f-cheer-01": ("Cheer", "Arms thrown up with a lean: large shoulder elevation and thoracic extension."),
    "f-claphands-01": ("Clap", "Hands brought together in front of the chest: scapular protraction."),
    "f-drink-drinking": ("Drink", "One arm lifts a cup to the mouth and the head tips back."),
    "f-gestic-presentation-left-01": ("Present to the left", "An open-handed gesture with a turn of the trunk and head."),
    "f-idle-shake-arms-01": ("Shake out the arms", "Loose, fast arm shaking with a bouncing trunk."),
}


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1000:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dst + ".part", "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(dst + ".part", dst)


def duration(path):
    from rocketbox_to_clip import load_glb, accessor
    js, b = load_glb(path)
    return max(float(accessor(js, b, s["input"]).max()) for s in js["animations"][0]["samplers"])


def run(cmd, env=None, log=None):
    e = dict(os.environ); e.update(env or {})
    with open(log, "a") if log else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"failed: {' '.join(cmd)} (see {log})")


def main():
    work, site = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or list(CLIPS)
    os.makedirs(f"{site}/media", exist_ok=True); os.makedirs(f"{site}/data", exist_ok=True); os.makedirs(work, exist_ok=True)
    mj = f"{site}/data/motions.json"; meta = {m["id"]: m for m in json.load(open(mj))} if os.path.exists(mj) else {}
    for n in names:
        if n in meta and os.path.exists(f"{site}/media/{n}.mp4"):
            print(n, "done already", flush=True); continue
        log = f"{work}/{n}.log"; print("==", n, flush=True)
        sk, eng = f"{work}/rocketbox/{n}.skeleton.glb", f"{work}/rocketbox/{n}.engine.glb"
        fetch(f"{HF}/skeleton/{n}/{n}.glb", sk); fetch(f"{HF}/engine/{n}/{n}.glb", eng)
        dur = duration(sk); stride = max(1, math.ceil((dur * 30 + 1) / TARGET_FRAMES))
        clip = f"{work}/clips/{n}.npz"; os.makedirs(f"{work}/clips", exist_ok=True)
        if not os.path.exists(clip):
            run([sys.executable, f"{ROOT}/tools/rocketbox_to_clip.py", sk, clip, "--mhr-onnx", os.environ["ONNX"], "--stride", str(stride)], log=log)
        z = np.load(clip); nf = len(z["mhr_model_params"]); err = float(z["direction_error_deg"].mean()); fps = float(z["fps"])
        frames = f"{work}/frames/{n}"
        have = len([f for f in os.listdir(frames) if f.endswith(".png")]) if os.path.isdir(frames) else 0
        if have < nf:
            todo = [i for i in range(nf) if not os.path.exists(f"{frames}/{i:03d}.png")]
            run([sys.executable, f"{ROOT}/experiments/render_clip.py", clip, frames, "--workers", "2", "--only", ",".join(map(str, todo))],
                env={"CTW_STEP": "3.0", "PYTHONPATH": f"{ROOT}:{ROOT}/tools"}, log=log)
        orig = f"{work}/with_orig/{n}"
        run([sys.executable, f"{ROOT}/tools/rocketbox_original_panel.py", eng, frames, orig, "--stride", str(stride)], log=log)
        run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "5", "-i", f"{orig}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", f"{site}/media/{n}.mp4"], log=log)
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{orig}/{nf // 2:03d}.png", "-vf", "scale=900:-1", "-q:v", "4", f"{site}/media/{n}.jpg"], log=log)
        title, blurb = CLIPS.get(n, (n, ""))
        meta[n] = {"id": n, "title": title, "blurb": blurb, "frames": nf, "fps": fps, "seconds": round(nf / fps, 1), "direction_error_deg": round(err, 1),
                   "source_url": f"https://huggingface.co/datasets/codepawl/rocketbox-motions-glb/tree/main/skeleton/{n}"}
        json.dump(list(meta.values()), open(mj, "w"), indent=1); print(n, "finished:", nf, "frames, mean direction error", round(err, 1), "deg", flush=True)


if __name__ == "__main__":
    main()
