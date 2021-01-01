# -*- coding: utf-8 -*-

import json
import glob

def json_concat(json_path, out_path):

    result = {}

    for f in glob.glob(json_path+"*.json"):
        with open(f, "rb") as infile:
            jsonFile = json.load(infile)
            keys = jsonFile.keys()
            for key in keys:
                currentProduct = jsonFile[key]
                result[key] = currentProduct#.encode()

    with open(out_path+"mergedFile.json", "wb") as outfile:
        json.dump(result, outfile)

if __name__ == '__main__':
    json_path = '/mnt/tangjinhui/10117_yanrui/dataset/sth/sth_else_staff/json_files/'
    out_path = '/mnt/tangjinhui/10117_yanrui/dataset/sth/sth_else_staff/'
    json_concat(json_path, out_path)