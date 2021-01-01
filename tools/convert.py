import pickle
import json

with open('dataset/bounding_box_annotations.json', 'rb') as f1:
    box_annotations = json.load(f1)

with open('dataset/bounding_box_annotations.pkl', 'wb') as f2:
    pickle.dump(box_annotations, f2, protocol=pickle.HIGHEST_PROTOCOL)