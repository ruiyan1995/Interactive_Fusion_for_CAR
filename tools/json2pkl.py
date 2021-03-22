import pickle
import json
  
### convert box anno from json to pkl for faster loading...

# json_pth = 'to_your_pth/sth_else_anno/SOMETHING_ELSE/compositional/bounding_box_smthsmth_part1.json'
json_pth = 'to_your_pth/sth_else_anno/SOMETHING_ELSE/detected_compositional/detected_compositional.json'

pkl_pth = json_pth.replace('json', 'pkl')

with open(json_pth, 'r') as json_file: 
    data = json.load(json_file)

with open(pkl_pth, 'wb') as pkl_file:
    pickle.dump(data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

### test pkl
with open(pkl_pth, 'rb') as pkl_file:
    pkl_data = pickle.load(pkl_file)
