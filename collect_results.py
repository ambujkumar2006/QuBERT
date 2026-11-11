"""
Copy only the small result files (history.json, test_predictions.npz) out of one or
more 'checkpoints' folders into a single bundle, optionally zipped.
The big last_checkpoint.pth files (~400 MB each for BERT) are NOT copied.

Examples
--------
    python collect_results.py --src /kaggle/working/classical/checkpoints --dst /kaggle/working/results
    python collect_results.py --src ./checkpoints --dst ./results --zip
"""
import argparse
import os
import shutil

FILES = ("history.json", "test_predictions.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", nargs="+", required=True, help="checkpoints folder(s)")
    parser.add_argument("--dst", required=True, help="output folder (a 'checkpoints' subfolder is created)")
    parser.add_argument("--zip", action="store_true", help="also create <dst>.zip")
    args = parser.parse_args()

    out = os.path.join(args.dst, "checkpoints")
    copied = 0

    for src in args.src:
        if not os.path.isdir(src):
            print(f"! {src} not found, skipping")
            continue
        for name in sorted(os.listdir(src)):
            for fname in FILES:
                path = os.path.join(src, name, fname)
                if os.path.isfile(path):
                    os.makedirs(os.path.join(out, name), exist_ok=True)
                    shutil.copy2(path, os.path.join(out, name, fname))
                    copied += 1
            if os.path.isdir(os.path.join(out, name)):
                print(f"collected {name}")

    print(f"{copied} files -> {out}")

    if args.zip:
        archive = shutil.make_archive(args.dst.rstrip("/"), "zip", args.dst)
        print(f"zipped -> {archive}")


if __name__ == "__main__":
    main()
