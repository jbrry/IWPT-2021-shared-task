import json
import logging
import collections
import math
import csv
import os
import sys


def main():

    # main log dir / results directory
    logdir = sys.argv[1]
    outdir = sys.argv[2]
    metric = sys.argv[3]

    KEYS = ["best_validation_upos_accuracy",
            "best_validation_xpos_accuracy",
            "best_validation_feats_accuracy",
            "best_validation_dependencies_LAS"]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    split_path = logdir.split("/")
    run = split_path[1]

    csv_file = open(f"{outdir}/{run}_{metric}_experiment_results.csv", "w")
    csv_writer = csv.writer(csv_file)

    d_completed = {}

    header = []
    header.append("run")
    for metric in KEYS:
        header.append(metric)
    csv_writer.writerow(header)

    for run in os.listdir(logdir):
        print(f"Run: {run} ")
        completed = False

        path = os.path.join(logdir, run)
        for _file in os.listdir(path):
            if _file == "metrics.json":    
                # Check for complete run
                completed = True

                results = {metric: None for metric in KEYS}
               
                with open(os.path.join(path, _file)) as json_file:
                    data = json.load(json_file)
                    for key in data:
                        if key in KEYS:
                            # metrics are not always deterministic so add to dictionary then
                            # write from there.
                            results[key] = data[key]

                    data_row = []
                    data_row.append(run)
                    for metric, score in results.items():
                        data_row.append(score)

                    csv_writer.writerow(data_row)

        d_completed[run] = completed
    print(d_completed)

if __name__ == '__main__':
    main()

