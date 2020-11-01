import os
import numpy as np
import pandas as pd

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(dir_path):
    summary_iterators = [EventAccumulator(os.path.join(dir_path, dname)).Reload() for dname in os.listdir(dir_path)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps = []

    for tag in tags:
        steps = [e.step for e in summary_iterators[0].Scalars(tag)]
        wall_times = [e.wall_time for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps, wall_times


def to_csv(log_dir_path, csv_dir_path):
    dirs = os.listdir(log_dir_path)

    d, steps, wall_times = tabulate_events(log_dir_path)
    tags, values = zip(*d.items())
    np_values = np.array(values)
    csv_columns = ['step', 'wall_time']
    csv_columns.extend(dirs)
    print('extend', ['step', 'wall_time'].extend(dirs))
    print('csv_columns', csv_columns)

    for index, tag in enumerate(tags):
        # df = pd.DataFrame(np_values[index], index=steps, columns=dirs)
        df = pd.DataFrame(np.vstack((steps, wall_times, np_values[index].T)).T, columns=csv_columns)
        df.to_csv(get_csv_file_path(csv_dir_path, tag), index=False)


def get_csv_file_path(csv_dir_path, tag):
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(csv_dir_path, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    # example
    train_id = 'SR_1D_CNN_SAMPLE-TRAIN'
    log_dir_path = "/var/tensorflow/tsp/sample/logs/{}/".format(train_id)
    csv_dir_path = "/var/tensorflow/tsp/sample/history/{}/".format(train_id)
    to_csv(log_dir_path, csv_dir_path)