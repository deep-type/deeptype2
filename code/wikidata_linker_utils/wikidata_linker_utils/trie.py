def has_keys_with_prefix(trie, prefix):
    for _ in trie.iterkeys(prefix=prefix):
        return True
    return False
