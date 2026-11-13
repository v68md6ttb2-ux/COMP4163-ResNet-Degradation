#!/usr/bin/env bash
# Resilient CIFAR-10 download.
#
# The Toronto server throttles each connection to a few tens of kB/s, which
# makes the single-stream torchvision download fail (it aborts on a stalled
# connection). This script fetches the same file with N parallel ranged
# requests, resumes each part from where it stopped, and only replaces the
# archive when the MD5 matches the official one.
#
# Usage:  bash scripts/fetch_cifar.sh [data-dir] [parts-dir] [url]
# The URL defaults to the official Toronto one; a mirror is fine because the
# assembled archive must match the official MD5 below either way.
set -u

URL=${3:-https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz}
SIZE=170498071
MD5=c58f30108f718f92721af3b95e74349a
N=24

ROOT=${1:-data}
PARTS=${2:-build/cifar-parts}
OUT="$ROOT/cifar-10-python.tar.gz"

mkdir -p "$ROOT" "$PARTS"
chunk=$(( (SIZE + N - 1) / N ))

for i in $(seq 0 $((N - 1))); do
  (
    start=$((i * chunk))
    end=$((start + chunk - 1))
    [ "$end" -ge "$SIZE" ] && end=$((SIZE - 1))
    want=$((end - start + 1))
    part=$(printf "%s/part_%02d" "$PARTS" "$i")
    for _ in $(seq 1 400); do
      have=$(stat -c %s "$part" 2>/dev/null || echo 0)
      [ "$have" -ge "$want" ] && break
      curl -sL --max-time 120 -r $((start + have))-$end "$URL" >> "$part" 2>/dev/null || true
    done
    have=$(stat -c %s "$part" 2>/dev/null || echo 0)
    if [ "$have" -gt "$want" ]; then truncate -s "$want" "$part"; fi
    [ "$have" -ge "$want" ] || echo "part $i incomplete: $have/$want" >&2
  ) &
done
wait

cat "$PARTS"/part_* > "$OUT.tmp"
actual=$(md5sum "$OUT.tmp" | cut -d' ' -f1)
if [ "$actual" = "$MD5" ]; then
  mv "$OUT.tmp" "$OUT"
  echo "ok: $OUT ($(stat -c %s "$OUT") bytes, md5 $actual)"
else
  echo "md5 mismatch: got $actual, expected $MD5" >&2
  echo "the archive was left at $OUT.tmp" >&2
  exit 1
fi
