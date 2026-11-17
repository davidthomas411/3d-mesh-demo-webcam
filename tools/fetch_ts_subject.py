"""Fetch ONE subject of the TotalSegmentator CT dataset (Wasserthal et al., Zenodo record 10047292, CC BY 4.0) without
downloading the 23.6 GB archive: the zip's directory is read with HTTP range requests and only the wanted subject's files are
fetched.

usage: python tools/fetch_ts_subject.py s1159 OUT_DIR   (needs: pip install remotezip)
       writes OUT_DIR/s1159/ct.nii.gz and OUT_DIR/s1159/segmentations/*.nii.gz"""
import os, sys, time

from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/10047292/files/Totalsegmentator_dataset_v201.zip/content"


def main():
    subject, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    with RemoteZip(URL, headers={"User-Agent": "python-remotezip (dataset fetch)"}) as z:
        names = [n for n in z.namelist() if subject in n.replace("\\", "/").split("/") and not n.endswith("/")]
        if not names:
            raise SystemExit(f"{subject} is not in the archive")
        total = sum(z.getinfo(n).compress_size for n in names)
        print(f"{len(names)} files of {subject}, {total / 1e6:.1f} MB compressed", flush=True)
        for i, n in enumerate(names):
            parts = n.replace("\\", "/").split("/")
            path = os.path.join(out, *parts[parts.index(subject):])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with z.open(n) as src, open(path, "wb") as dst:
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    dst.write(b)
            if i % 20 == 0:
                print(f"  {i + 1}/{len(names)} {parts[-1]}", flush=True)
    print(f"DONE {len(names)} files in {(time.time() - t0) / 60:.1f} min -> {os.path.join(out, subject)}", flush=True)


if __name__ == "__main__":
    main()
