import json
import os

import zipfile

from tqdm import tqdm

if __name__ == '__main__':

    # I/O
    dataset_base_path = '/home/felix/repos/LID-DS/LID-DS-2021/'
    target_dir = '/media/felix/PortableSSD/LID-DS-2021-no-relative-time'

    # listings
    categories = ['training', 'test', 'validation']
    subcategories = ['normal', 'normal_and_attack']
    times = ['container_ready', 'warmup_end']

    """
    for scenario in tqdm(SCENARIO_NAMES):
        dataloader = DataLoader(os.path.join(dataset_base_path, scenario))
        for category in tqdm(categories):
            for recording in dataloader.extract_recordings(category):
                recording_path = recording.path
                sub_path = recording.path.replace(dataset_base_path, '')
                src_zip = recording_path
                dst_zip = os.path.join(target_dir, sub_path)
                recording_name = os.path.splitext(os.path.basename(os.path.normpath(recording_path)))[0]
                scenario_name = sub_path.split('/')[0]

                # creating dataset directories
                try:
                    os.mkdir(os.path.join(target_dir, scenario_name))
                except FileExistsError:
                    pass
                for dir_name in categories:
                    try:
                        os.mkdir(os.path.join(target_dir, scenario_name, dir_name))
                    except FileExistsError:
                        pass
                for dir_name in subcategories:
                    try:
                        os.mkdir(os.path.join(target_dir, scenario_name, 'test', dir_name))
                    except FileExistsError:
                        pass
                
                """
    """
    # reading from original zip, adjusting data and compressing it as new version
    with zipfile.ZipFile(src_zip) as inzip, zipfile.ZipFile(dst_zip, "w",
                                                            zipfile.ZIP_DEFLATED,
                                                            compresslevel=8) as outzip:
        for inzipinfo in inzip.infolist():
            with inzip.open(inzipinfo) as infile:
                if inzipinfo.filename == f'{recording_name}.json':
                    # removing relative timestamp
                    content = json.loads(infile.read().decode('utf-8'))
                    for time in times:
                        del content['time'][time]['relative']
                    for exploit in content['time']['exploit']:
                        del exploit['relative']
                    outzip.writestr(inzipinfo.filename, json.dumps(content, indent=4))

                # special case for ZipSlip scenario that contains process names with spaces
                elif scenario == 'ZipSlip' and inzipinfo.filename == f'{recording_name}.sc':
                    content = infile.read()
                    # replacing spaces with '_'
                    content = content.replace(b'C2 CompilerThre', b'C2_CompilerThre')
                    content = content.replace(b'C1 CompilerThre', b'C1_CompilerThre')

                    outzip.writestr(inzipinfo.filename, content)
                else:
                    content = infile.read()
                    outzip.writestr(inzipinfo.filename, content)
    
    """

    with open('/home/felix/repos/LID-DS/tools/timestamp_failures/false_timestamp_sequences.json') as json_file:
        data = json.load(json_file)
        i = 0
        path_set = set()
        for scenario in data.keys():
            for data_part in data[scenario].keys():
                for recording_type in data[scenario][data_part].keys():
                    for entry in data[scenario][data_part][recording_type]:
                        i += 1
                        src_zip = entry['recording']
                        path_set.add(src_zip)

    for src_zip in tqdm(path_set):


        dst_zip = src_zip.replace('/home/felix/datasets/LID-DS-2021/',
                                  '/media/felix/PortableSSD/rerecord/LID-DS-2021-sorted/')

        with zipfile.ZipFile(src_zip) as inzip, zipfile.ZipFile(dst_zip, "w",
                                                                zipfile.ZIP_DEFLATED,
                                                                compresslevel=8) as outzip:
            for inzipinfo in inzip.infolist():
                with inzip.open(inzipinfo) as infile:
                    if inzipinfo.filename.endswith('.sc'):
                        content = infile.readlines()
                        content.sort()
                        new_content = ''
                        for line in content:
                            new_content += line.decode()
                        outzip.writestr(inzipinfo.filename, new_content)
                    else:
                        content = infile.read()
                        outzip.writestr(inzipinfo.filename, content)
