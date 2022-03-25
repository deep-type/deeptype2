from os.path import dirname, realpath, join, isfile
from os import listdir
import json

SCRIPT_DIR = dirname(realpath(__file__))

def main():
    fnames = listdir(SCRIPT_DIR)
    fnames = [join(SCRIPT_DIR, fname)  
              for fname in fnames if isfile(join(SCRIPT_DIR, fname)) and fname.endswith(".json") and "anonymized" not in fname]
    all_names = set()
    all_hit_ids = set()
    all_data = []
    for fname in fnames:
        with open(fname, "rt") as fin:
            all_data.append(json.load(fin))
    for data in all_data:
        for row in data:
            for worker in row["worker_ids"]:
                if worker not in all_names:
                    all_names.add(worker)
            if row["hit_id"] not in all_hit_ids:
                all_hit_ids.add(row["hit_id"])
    name_mapping = {name: f"participant_{idx}" for idx, name in enumerate(sorted(all_names))}
    hit_mapping = {hit: f"hit_{idx}" for idx, hit in enumerate(sorted(all_hit_ids))}

    for data, fname in zip(all_data, fnames):
        new_fname = fname.split(".json", 1)[0] + "_anonymized.json"
        for row in data:
            row["worker_ids"] = [name_mapping[name] for name in row["worker_ids"]]
            row["hit_id"] = hit_mapping[row["hit_id"]]
        with open(new_fname, "wt") as fout:
            json.dump(data, fout, separators=(',', ':'), indent=None)

if __name__ == "__main__":
    main()
