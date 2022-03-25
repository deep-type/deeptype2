def nested_hash_vals(hasher, val, ignore_keys=None):
    if isinstance(val, dict):
        for key in sorted(val.keys()):
            if ignore_keys is not None and key in ignore_keys:
                continue
            hasher.update(key.encode("utf-8"))
            nested_hash_vals(hasher, val[key], ignore_keys)
    elif isinstance(val, (list, tuple)):
        for subval in val:
            nested_hash_vals(hasher, subval, ignore_keys)
    elif isinstance(val, str):
        hasher.update(val.encode("utf-8"))
    else:
        hasher.update(str(val).encode("utf-8"))
