import pandas as pd
from annoy import AnnoyIndex
from fuzzywuzzy import fuzz
import numpy as np
import configs

image_data = pd.read_pickle(configs.image_data_with_features_pkl)

f = len(image_data['features'][0])
def search_by_vector(v):
    u = AnnoyIndex(f, 'euclidean')
    u.load(configs.image_features_vectors_ann) # super fast, will just mmap the file
    index_list = u.get_nns_by_vector(v, 4) # will find the 10 nearest neighbors
    return image_data.iloc[index_list]['local_path'].to_list(),image_data.iloc[index_list]['titles'].to_list()

def get_similar_text(text:str):
    matching = []
    for i in image_data['titles']:
        matching.append(fuzz.token_set_ratio(text,i))
    #return data['features'][np.argmax(matching)]
    index_list = np.argsort(matching,)[-4:][::-1]
    return image_data.iloc[index_list]['local_path'].to_list(),image_data.iloc[index_list]['titles'].to_list()

def Hybrid_Search(text:str):
    matching = []
    for i in image_data['titles']:
        matching.append(fuzz.token_set_ratio(text,i))
    return image_data['features'][np.argmax(matching)]

