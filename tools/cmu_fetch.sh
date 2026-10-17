#!/usr/bin/env bash
# Fetch the human motion reference for the imitation channel.
#
# Source data: CMU Graphics Lab Motion Capture Database. Its FAQ
# (http://mocap.cs.cmu.edu/faqs.php) states, verbatim: "The motion capture data
# may be copied, modified, or redistributed without permission", and the homepage
# says "This dataset of motions is free for all uses". That is why the motion
# reference comes from here and not from Mixamo: Adobe's General Terms section
# 3.6 grant use inside a product you author but forbid distributing Content Files
# "on a stand-alone basis, outside of the End Use".
#
# CMU publishes ASF/AMC only, and its Vicon-dialect pair leaves the bone-axis
# convention ambiguous (an FK built from it keeps the elbow deflection pinned at
# 90 deg through a whole punch trial -- see tools/bvh_to_dirs.py's docstring).
# The working input is therefore a BVH conversion of the SAME trials, from a
# community repository that carries no licence file of its own: the underlying
# motion remains CMU's, a format conversion adds no authorship, and we vendor
# none of that repository's code -- only the data, so the citation to CMU above
# is the one that matters. Both files are kept locally; nothing here is
# committed (runs/ is git-ignored).
#
# CMU's requested citation (READMEFIRST.txt): "The data used in this project was
# obtained from mocap.cs.cmu.edu. The database was created with funding from NSF
# EIA-0196217."
#
#   bash tools/cmu_fetch.sh
set -euo pipefail

DEST="$(dirname "$0")/../runs/reference/cmu"
mkdir -p "$DEST"

# Original CMU recordings: subject 144 is a punching session, subject 13 boxing.
for trial in 144/144.asf 144/144_13.amc 13/13.asf 13/13_17.amc 13/13_18.amc; do
    name=$(basename "$trial")
    [ -s "$DEST/$name" ] && { echo "have    $name"; continue; }
    if curl -sS --max-time 120 -o "$DEST/$name" \
        "http://mocap.cs.cmu.edu/subjects/$trial" && [ -s "$DEST/$name" ]; then
        echo "got     $name ($(wc -c < "$DEST/$name") bytes)"
    else
        echo "missing $trial"; rm -f "$DEST/$name"
    fi
done

# BVH conversions of the same trials, used by tools/bvh_to_dirs.py.
for bvh in 13_17 13_18; do
    zip="$DEST/$bvh.zip"
    if [ ! -s "$zip" ]; then
        if curl -sSL --max-time 180 -o "$zip" \
            "https://raw.githubusercontent.com/Shriinivas/cmubvh/main/Sequence-010-014/13/Data/$bvh.zip" \
            && [ -s "$zip" ]; then
            echo "got     $bvh.zip ($(wc -c < "$zip") bytes)"
        else
            echo "missing $bvh.zip"; rm -f "$zip"
        fi
    else
        echo "have    $bvh.zip"
    fi
    if [ -s "$zip" ] && [ ! -s "$DEST/$bvh.bvh" ]; then
        python3 -c "
import sys, zipfile
name = '$bvh'
with zipfile.ZipFile('$DEST/' + name + '.zip') as archive:
    open('$DEST/' + name + '.bvh', 'wb').write(archive.read(archive.namelist()[0]))
print('unpacked', name + '.bvh')
"
    fi
done
