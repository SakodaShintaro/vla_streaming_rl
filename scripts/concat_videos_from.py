import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", type=Path)
    parser.add_argument("start_filename", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    all_files = sorted(args.video_dir.glob("best_*.mp4"))
    target_files = [f for f in all_files if f.name >= args.start_filename]
    output_path = args.video_dir / f"{Path(args.start_filename).stem}_concat.mp4"

    if len(target_files) == 0:
        print(f"No files found from '{args.start_filename}' onward in {args.video_dir}")
        sys.exit(1)

    print(f"Concatenating {len(target_files)} files:")
    for f in target_files:
        print(f"  {f.name}")

    list_path = output_path.parent / f"{output_path.stem}_filelist.txt"
    with open(list_path, "w") as f:
        for video_file in target_files:
            f.write(f"file '{video_file.resolve()}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ],
        check=True,
    )

    list_path.unlink()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
