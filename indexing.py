# Import the libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
import configs

class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)

def get_feature(image_data:list):
    # Iterate through images (Change the path based on your image location)
    fe = FeatureExtractor()
    features = []
    for img_path in tqdm(image_data):
        #print(img_path)
        # Extract Features
        try:
            feature = fe.extract(img=Image.open(img_path))
            features.append(feature)
        except:
            features.append(None)
            continue
    return features

def start_feature_extraction():
    image_data = pd.read_csv(configs.image_data_csv)
    f_data = get_feature(image_data['local_path'].to_list())
    image_data['features']  = f_data
    return image_data.dropna().reset_index(drop=True)

def start_indexing(image_data):
    f = len(image_data['features'][0]) # Length of item vector that will be indexed
    t = AnnoyIndex(f, 'euclidean')
    for i,v in tqdm(zip(image_data.index,image_data['features'])):
        t.add_item(i, v)
    t.build(100) # 100 trees
    print("Saved the Indexed File:"+"[image_features_vectors.ann]")
    t.save(configs.image_features_vectors_ann)

def main():
    print("___Features Extraction Started____")
    image_data = start_feature_extraction()
    print("___Features Extraction Done____")

    print("\n___Feature Data Indexing Started____")
    start_indexing(image_data)
    print("\n___Feature Data Indexing Done____")

    print("\n___Image MetaData Saved: [image_data_features.pkl]___")
    image_data.to_pickle(configs.image_data_with_features_pkl)

if __name__== "__main__":
    main()
