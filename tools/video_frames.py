"""Tile evenly spaced frames of a video into one PNG contact sheet."""
import argparse

import imageio.v2 as imageio
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("output")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--start", type=float, default=0.0, help="seconds")
    parser.add_argument("--end", type=float, help="seconds")
    args = parser.parse_args()
    reader = imageio.get_reader(args.video)
    fps = reader.get_meta_data()["fps"]
    frames = [np.asarray(f) for f in reader]
    first = int(args.start * fps)
    last = min(len(frames), int(args.end * fps)) if args.end else len(frames)
    picks = np.linspace(first, last - 1, args.count).astype(int)
    tiles = [frames[i] for i in picks]
    rows = [np.concatenate(tiles[i:i + args.columns], axis=1)
            for i in range(0, len(tiles), args.columns)]
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    imageio.imwrite(args.output, np.concatenate(rows, axis=0))
    print(f"{len(frames)} frames at {fps} fps; picked {picks.tolist()}")


if __name__ == "__main__":
    main()
