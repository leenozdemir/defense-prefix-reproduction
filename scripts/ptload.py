"""Minimal torch-free reader for torch.save() zip files (enough for our cached-feature dicts)."""
import pickle
import zipfile

import numpy as np

_DTYPES = {
    "HalfStorage": np.float16, "FloatStorage": np.float32, "DoubleStorage": np.float64,
    "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_,
    "ByteStorage": np.uint8, "BFloat16Storage": np.uint16,
}


class _StorageType:
    def __init__(self, name):
        self.name = name


def _rebuild_tensor_v2(storage, offset, size, stride, *args, **kwargs):
    if len(size) == 0:
        return storage[offset]
    itemsize = storage.dtype.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)).copy()


def load(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    prefix = names[0].split("/")[0]
    cache = {}

    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "torch" and name.endswith("Storage"):
                return _StorageType(name)
            if module == "collections" and name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if module == "torch.torch_version" and name == "TorchVersion":
                return str
            if module == "torch._utils" and name == "_rebuild_parameter":
                return lambda data, *a, **k: data
            return super().find_class(module, name)

        def persistent_load(self, pid):
            _, stype, key, _loc, _numel = pid
            if key not in cache:
                raw = zf.read(f"{prefix}/data/{key}")
                cache[key] = np.frombuffer(raw, dtype=_DTYPES[stype.name])
            return cache[key]

    return U(zf.open(f"{prefix}/data.pkl")).load()
