import numpy as np
import json
from collections import defaultdict
import argparse
import csv
import os


def parse_option():
    parser = argparse.ArgumentParser('argument for predictions')
    parser.add_argument('--json_dir', type=str, default='./jsons/')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--csv_name', type=str)
    opt = parser.parse_args()
    return opt


def create_csv(opt):
    for entry in os.scandir(opt.json_dir):
        if entry.path.endswith('.json'):
            print(entry.path)
            with open(entry.path, 'r') as json_file:
                data = []
                matrix = json.load(json_file)
                matrix = np.array(matrix)
                for i in range(15):
                    obs = defaultdict(list)
                    future = defaultdict(list)
                    for j in range(1, 61, 2):
                        poses = defaultdict(list)
                        frame = matrix[matrix[:, 0] == i * 30 + j]
                        for pose in frame:
                            for kp_number in range(5, 8):
                                poses[pose[1]].append(pose[kp_number])

                        for p_id in poses.keys():
                            if j < 32:
                                obs[p_id].append(poses[p_id])
                            else:
                                future[p_id].append(poses[p_id])
                    for p_id in obs:
                        if p_id in future.keys() and obs[p_id].__len__() == 16 and future[p_id].__len__() == 14:
                            data.append([p_id, obs[p_id], future[p_id]])
            with open(os.path.join(opt.output_dir, opt.csv_name), 'a') as f_object:
                writer = csv.writer(f_object)
                writer.writerows(data)


if __name__ == '__main__':
    opt = parse_option()
    header = ['pedestrian_id', 'observed_pose', 'future_pose']
    with open(os.path.join(opt.output_dir, opt.csv_name), 'a') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(header)
    create_csv(opt)
