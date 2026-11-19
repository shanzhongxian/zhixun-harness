import hashlib
import sys
import tarfile

path = sys.argv[1]
checked = 0
bad = []
with tarfile.open(path) as tf:
    for item in tf:
        if not item.isfile() or not item.name.startswith("blobs/sha256/"):
            continue
        expected = item.name.rsplit("/", 1)[1]
        digest = hashlib.sha256()
        handle = tf.extractfile(item)
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
        checked += 1
        if digest.hexdigest() != expected:
            bad.append((item.name, digest.hexdigest()))
print("blobs_checked=%d mismatch=%d" % (checked, len(bad)))
for name, got in bad:
    print("BAD", name, got)
