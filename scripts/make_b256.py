import numpy as np, os
c = os.environ.get("STPP_CACHE_DIR", "/mnt/f/aiwork/mixed_cache_big")
for split in ("train", "test"):
    src = f"{c}/{split}_b512.npy"
    dst = f"{c}/{split}_b256.npy"
    a = np.load(src)
    assert a.shape[1] == 512, a.shape
    b = a.reshape(a.shape[0], 2, 256).reshape(-1, 256)
    np.save(dst, b)
    print(split, a.shape, "->", b.shape)