import json
import math
from argparse import ArgumentParser
from mpi4py import MPI

import numpy as np
import os


def parse_args():
    parser = ArgumentParser('Process files')
    parser.add_argument('--mode', required=True, choices=['sequence', 'parallel'])

    return parser.parse_args()


def read_files(from_file: int, to_file: int):
    """
        Read data in files in interval [from_file, to_file)
        Outputs np.array of data files in every interval
    """

    batch_values = []
    if from_file != -1 and  to_file != -1:
        for index in range(from_file, to_file + 1):
            # Read values from every file and append values to batch_values

            # Sasha
            with open('content/files/' + str(index) + '.txt', 'r', encoding="utf-8") as f:
                for one_line in f:
                    batch_values.append(float(one_line.strip()))
                    # Sasha, работает и без strip, какое должно быть имя файла - index?
            # pass

    # return np.concatenate(batch_values)
    return batch_values


def calculate_statistics(client_sums):
    sums = sum(client_sums)
    # sum_squares = sum([i * i for i in client_sums])
    count = len(client_sums)
    sum_squares = sum([(i - (sums/count)) ** 2 for i in client_sums])

    below_1M = len([i for i in client_sums if i < 1000])
    count_1M_10M = len([i for i in client_sums if 1000 < i < 10000])
    count_10M_100M = len([i for i in client_sums if 10000 < i < 100000])
    count_100M_1B = len([i for i in client_sums if 100000 < i < 1000000])
    more_1B = len([i for i in client_sums if i > 1000000])

    return {
        'count': count,
        '1M': below_1M,
        '10M': count_1M_10M,
        '100M': count_10M_100M,
        '1B': count_100M_1B,
        '1B+': more_1B,
        'sum': sums,
        'sum_squares': sum_squares
    }


def accumulate_stats(stats):
    all_count = sum(item['count'] for item in stats)
    all_sums = sum(item['sum'] for item in stats)
    all_sums_squared = sum(item['sum_squares'] for item in stats)

    avg = all_sums / all_count
    var = all_sums_squared / all_count

    results = {
        'avg': avg,
        'std': np.sqrt(var),
        'hist': {}
    }

    for key in ['1M', '10M', '100M', '1B', '1B+']:
        # Calculate sum of values by stats
        result = sum(item[key] for item in stats)
        results['hist'][key] = result

    return results


def parallel_pipeline():
    comm = MPI.COMM_WORLD
    world_size = comm.Get_size()
    world_rank = comm.Get_rank()

    splt_fl = []

    if world_rank == 0:
        files = os.listdir('content/files')
        num_files = len(files)
        if num_files == 0:
            raise Exception('Нет файлов!')
        files.sort()
        part_fl = math.ceil(num_files / world_size)
        splt_fl = np.arange(world_size * part_fl)
        for x in splt_fl:
            if str(x) + '.txt' not in files:
                splt_fl[x] = -1
        splt_fl.shape = world_size, part_fl
    else:
        splt_fl = None
    splt_fl = comm.scatter(splt_fl, root=0)

    end_file = -1
    start_file = splt_fl[0]
    for i in reversed(splt_fl):
        if i != -1:
            end_file = i
            break

    data = read_files(start_file, end_file)
    stats = calculate_statistics(data)

    # Get all stats via gather
    received_stats = comm.gather(stats, root=0)
    if world_rank == 0:
        return accumulate_stats(received_stats)
    else:
        return None


def sequence_pipeline():
    # Run sequence pipeline like parallel pipeline - but without MPI
    files = os.listdir('content/files')
    num_files = len(files)
    if num_files == 0:
        raise Exception('Нет файлов!')
    files.sort()
    start_file = int(str.replace(files[0], '.txt', ''))
    end_file = int(str.replace(files[-1], '.txt', ''))
    data = read_files(start_file, end_file)
    stats = calculate_statistics(data)
    return accumulate_stats([stats])


def main():
    args = parse_args()
    if args.mode == 'parallel':
        world_rank = MPI.COMM_WORLD.Get_rank()
        output = parallel_pipeline()
        if world_rank == 0:
            print(output)
            if output is not None:
                with open("output.json", "w") as fp:
                    json.dump(output, fp, indent=4, ensure_ascii=False)
    else:
        world_rank = MPI.COMM_WORLD.Get_rank()
        if world_rank == 0:
            output = sequence_pipeline()
            print(output)
            if output is not None:
                with open("output.json", "w") as fp:
                    json.dump(output, fp, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    main()