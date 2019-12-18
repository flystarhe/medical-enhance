import os
import sys
import json
import shutil
from pathlib import Path
import SimpleITK as sitk
from collections import defaultdict

'''
0008|0060 - Modality(MRI/CT/CR/DR)
0008|0070 - Manufacturer
0018|0015 - Body Part Examined
'''

G_TAGS = ['0008|0060', '0008|0070', '0018|0015']


def get_series_ids(data_dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(data_dir)
    return series_ids


def get_series_files(data_dir, series_id):
    reader = sitk.ImageSeriesReader()
    series_files = reader.GetGDCMSeriesFileNames(data_dir, series_id)
    return series_files


def meta_data_from_file(dcm_path):
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(dcm_path)
    file_reader.ReadImageInformation()
    return [file_reader.GetMetaData(tag).strip() for tag in G_TAGS]


def copy_files(file_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file_path in file_list:
        if file_path[-4:] != '.dcm':
            filename = os.path.basename(file_path) + '.dcm'
            shutil.copy(file_path, os.path.join(output_dir, filename))
        else:
            shutil.copy(file_path, output_dir)


def split_dir(input_dir, output_dir):
    cache = Path(input_dir).glob('**/*')
    cache = set([i.parent.as_posix() for i in cache])

    shutil.rmtree(output_dir, ignore_errors=True)

    counts = defaultdict(int)
    for data_dir in sorted(cache):
        for series_id in get_series_ids(data_dir):
            dcm_list = get_series_files(data_dir, series_id)
            if len(dcm_list) < 3:
                continue

            sub_dir = '_'.join(meta_data_from_file(dcm_list[0]))
            copy_files(dcm_list, os.path.join(output_dir, sub_dir, series_id))
            counts[sub_dir] += 1

    msg = ['# ReadMe', '', '## counts', json.dumps(counts, indent=4)]
    with open(os.path.join(output_dir, 'readme.md')) as f:
        f.write('\n'.join(msg))
    print('\n'.join(msg))


if __name__ == '__main__':
    split_dir(sys.argv[1], sys.argv[2])
