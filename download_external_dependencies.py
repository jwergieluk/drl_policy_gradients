import os
import requests
import zipfile
import stat


URLS = [
    # "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip",
    # "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip"
    "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip",
    "https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip"
    ]


if __name__ == '__main__':
    for url in URLS:
        file_name = url.split('/')[-1]
        if os.path.isfile(file_name):
            continue

        print('Downloading ' + url)
        response = requests.get(url)
        response.raise_for_status()

        with open(file_name, mode='bw') as f:
            f.write(response.content)
            print('Saved ' + file_name)

        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall('.')

        dir_name = file_name.split('.')[0]
        exec_file = os.path.join(dir_name, 'Reacher.x86_64')
        os.chmod(exec_file, os.stat(exec_file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
