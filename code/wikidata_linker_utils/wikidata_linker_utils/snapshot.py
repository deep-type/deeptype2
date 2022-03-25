import sys
import datetime
import shutil
from os.path import join, dirname, realpath
from os import makedirs

WIKIDATA_LINKER_UTILS_DIR = dirname(dirname(dirname(realpath(__file__))))

def snapshot_code_and_commands(save_dir):
    date = datetime.datetime.now().isoformat().replace(":", "_").split(".")[0]
    code_path = join(save_dir, f"code_{date}")
    makedirs(code_path, exist_ok=True)
    print(f"Saving code and commands to save directory under: {code_path}")
    with open(join(code_path, "launch.sh"), "wt") as fout:
        fout.write(" ".join(sys.argv) + "\n")
    shutil.copytree(WIKIDATA_LINKER_UTILS_DIR, join(code_path, "wikidata_linker_utils"),
                    ignore=shutil.ignore_patterns("ud-treebank*",
                                                  "*.pkl",
                                                  ".git",
                                                  ".ipynb_checkpoints",
                                                  ".pytest_cache",
                                                  "*.ipynb",
                                                  "scenario_data",
                                                  "__pycache__"))
