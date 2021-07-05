# adapted from: https://github.com/Hyperparticle/udify/blob/master/concat_treebanks.py

"""
Concatenates treebanks together
"""
from typing import List, Tuple, Dict, Any

import os
import shutil
import logging
import argparse



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("output_dir", type=str, help="The path to output the concatenated files")
parser.add_argument("--dataset_dir", default="data/train-dev", type=str,
                    help="The path containing all UD treebanks")
parser.add_argument("--treebanks", default=[], type=str, nargs="+",
                    help="Specify a list of treebanks to use; leave blank to default to all treebanks available")

args = parser.parse_args()


def get_ud_treebank_files(dataset_dir: str, treebanks: List[str] = None) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieves all treebank data paths in the given directory.
    :param dataset_dir: the directory where all treebank directories are stored
    :param treebanks: if not None or empty, retrieve just the subset of treebanks listed here
    :return: a dictionary mapping a treebank name to a list of train, dev, and test conllu files
    """
    datasets = {}
    treebanks = os.listdir(dataset_dir) if not treebanks else treebanks
    
    for treebank in treebanks:
        treebank_path = os.path.join(dataset_dir, treebank)
        conllu_files = [file for file in sorted(os.listdir(treebank_path)) if file.endswith(".conllu")]

        train_file = [file for file in conllu_files if file.endswith("train.conllu")]
        dev_file = [file for file in conllu_files if file.endswith("dev.conllu")]
        #test_file = [file for file in conllu_files if file.endswith("test.conllu")]

        train_file = os.path.join(treebank_path, train_file[0]) if train_file else None
        dev_file = os.path.join(treebank_path, dev_file[0]) if dev_file else None
        #test_file = os.path.join(treebank_path, test_file[0]) if test_file else None

        datasets[treebank] = (train_file, dev_file)

    return datasets

treebanks = get_ud_treebank_files(args.dataset_dir, args.treebanks)

# main treebank is the first one;
# where we will concatenate the other ones to that
main_treebank = args.treebanks[0]
main_treebank_files = treebanks[main_treebank]

output_path = os.path.join(args.output_dir, main_treebank)
if not os.path.exists(output_path):
    print(f"Creating output path {output_path}")
    os.makedirs(output_path)

basenames = []
for file_path in main_treebank_files:
    basename = os.path.basename(file_path)
    basenames.append(basename)

train, dev = list(zip(*[treebanks[k] for k in treebanks]))

print(f"train: {train}")
print(f"dev: {dev}")

for treebank, name in zip([train, dev ], basenames):
    with open(os.path.join(output_path, name), 'w') as write:
        for t in treebank:
            if not t:
                continue
            with open(t, 'r') as read:
                shutil.copyfileobj(read, write)