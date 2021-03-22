# -*- coding: utf-8 -*-

import json
import glob

def json_concat(json_path, out_path):

    result = {}

    for f in glob.glob(json_path+"*part*.json"):
        with open(f, "r") as infile:
            print(infile)
            jsonFile = json.load(infile)
            keys = jsonFile.keys()
            for key in keys:
                currentProduct = jsonFile[key]
                # currentProduct = [str(x).encode() for x in currentProduct]
                result[key] = currentProduct#.encode("utf-8")

    with open(out_path+"mergedFile.json", "w") as outfile:
        json.dump(result, outfile)

if __name__ == '__main__':
    json_path = 'to_your_pth/sth_else_anno/SOMETHING_ELSE/detected_fewshot/'
    out_path = 'to_your_pth/sth_else_anno/SOMETHING_ELSE/detected_fewshot/'
    json_concat(json_path, out_path)
    # Rename the merged file yourself