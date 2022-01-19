import json
import argparse
import os
from tqdm import tqdm

from dataloader.data_loader_2021 import DataLoader2021 as DataLoader, RecordingType
from dataloader.recording_2021 import Recording2021
from dataloader.syscall_2021 import Syscall2021

SCENARIO_NAMES = [
    "Bruteforce_CWE-307",
    "CVE-2012-2122",
    "CVE-2014-0160",
    "CVE-2017-7529",
    "CVE-2017-12635_6",
    "CVE-2018-3760",
    "CVE-2019-5418",
    "CVE-2020-9484",
    "CVE-2020-13942",
    "CVE-2020-23839",
    "CWE-89-SQL-injection",
    "EPS_CWE-434",
    "Juice-Shop",
    "PHP_CWE-434",
    "ZipSlip"
]

def save_to_json(results: dict, output_path: str):
    """

    saves results for one scenario to json file located at a given path
    overwrites old files

    """
    with open(os.path.join(output_path, 'false_timestamp_sequences.json'), 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)



def find_wrong_timestamps(recording_list: list, description: str):
    """

        finds wrong timestamp sequences

    """
    result = []
    for recording in tqdm(recording_list, description, unit=" recordings", smoothing=0):
        syscalls = recording.syscalls()
        recording: Recording2021
        previous_syscall = None
        for syscall in syscalls:
            syscall: Syscall2021
            if previous_syscall is None:
                previous_syscall = syscall
            else:
                if syscall.timestamp_unix_in_ns() < previous_syscall.timestamp_unix_in_ns():
                    if syscall.thread_id() != previous_syscall.thread_id():
                        thread_change = True
                    else:
                        thread_change = False

                    result.append({
                        'recording': recording.path,
                        'previous_line_number': previous_syscall.syscall_line,
                        'syscall_line_number': syscall.syscall_line,
                        'thread_change': thread_change,
                        'previous_syscall': previous_syscall.name(),
                        'syscall': syscall.name(),
                        'syscall_line_id': syscall.line_id,

                    })
            previous_syscall = syscall

    return result



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detection Tool to find empty records in LID-DS 2021')

    parser.add_argument('-d', dest='base_path', action='store', type=str, required=True,
                        help='LID-DS Base Path')
    parser.add_argument('-o', dest='output_path', action='store', type=str, required=True,
                        help='Output Path for statistics')

    args = parser.parse_args()

    result_dict = {}

    # iterates through list of all scenarios, main loop
    for scenario in SCENARIO_NAMES:

        scenario_path = os.path.join(args.base_path, scenario)
        dataloader = DataLoader(scenario_path)

        # dict to describe dataset structure
        data_parts = {
            'Training': {
                'Idle': dataloader.training_data(recording_type=RecordingType.IDLE),
                'Normal': dataloader.training_data(recording_type=RecordingType.NORMAL)
            },
            'Validation': {
                'Idle': dataloader.validation_data(recording_type=RecordingType.IDLE),
                'Normal': dataloader.validation_data(recording_type=RecordingType.NORMAL)
            },
            'Test': {
                'Idle': dataloader.test_data(recording_type=RecordingType.IDLE),
                'Normal': dataloader.test_data(recording_type=RecordingType.NORMAL),
                'Attack': dataloader.test_data(recording_type=RecordingType.ATTACK),
                'Normal and Attack': dataloader.test_data(recording_type=RecordingType.NORMAL_AND_ATTACK)
            }
        }

        # runs calculation for every recording type of every data part in data_part dictionary
        for data_part in data_parts.keys():
            for recording_type in data_parts[data_part].keys():
                result = find_wrong_timestamps(data_parts[data_part][recording_type],
                                               f"{scenario}: {data_part} - {recording_type}".rjust(45))

                if scenario not in result_dict.keys():
                    result_dict[scenario] = {}

                if data_part not in result_dict[scenario].keys():
                    result_dict[scenario][data_part] = {}

                result_dict[scenario][data_part][recording_type] = result

    save_to_json(result_dict, args.output_path)
