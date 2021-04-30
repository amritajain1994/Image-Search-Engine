import os

download_list = ['Animals/Arachnids','Animals/Birds','Animals/Fish','Animals/Insects','Animals/Mammals','Animals/Reptiles'] \
                + ['Plants/Flowers','Plants/Fruits'] + ['Places/Architecture','Places/Landscapes','Places/Panorama','Places/Urban']

wiki_url = 'https://en.m.wikipedia.org/wiki/Wikipedia:Featured_pictures/'

image_folder = 'wiki-images'

image_data_csv = os.path.join('saved-files/','image_data.csv')

image_data_with_features_pkl = os.path.join('saved-files/','image_data_features.pkl')

image_features_vectors_ann = os.path.join('saved-files/','image_features_vectors.ann')