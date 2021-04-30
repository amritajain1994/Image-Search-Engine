import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import searching
from indexing import FeatureExtractor

st.title("AI Image Serach Engine")
s_type  = st.selectbox('Select Serach Type:', ["Text Based Search","Image Based Search","Hybrid Search"])
if s_type == 'Image Based Search':
    search_image = st.file_uploader("Choose an image...", type="jpg")
    if search_image is not None:
        st.subheader("Your Sraching for following image:")
        img = Image.open(search_image)
        st.image(img, channels="BGR",width=100)
        fe = FeatureExtractor()
        query_vector = fe.extract(img)
        img_list,img_name = searching.search_by_vector(query_vector)
        st.subheader("Similar Result Found:")
        # Visualize the result
        axes=[]
        fig=plt.figure(figsize=(20,20))
        for a in range(2*2):
            #score = scores[a]
            axes.append(fig.add_subplot(2, 2, a+1))
            subplot_title=str(img_name[a])
            axes[-1].set_title(subplot_title)  
            plt.axis('off')
            plt.imshow(Image.open(img_list[a]))
        fig.tight_layout()
        st.write(fig)
elif s_type == 'Text Based Search':
        query_text = st.text_input('Enter some text for search: bear,tower,city etc.')
        if query_text:
            img_list,img_name = searching.get_similar_text(query_text)
            st.subheader(f"Similar Result Found For Keyword: {query_text}")
            # Visualize the result
            axes=[]
            fig=plt.figure(figsize=(20,20))
            for a in range(2*2):
                #score = scores[a]
                axes.append(fig.add_subplot(2, 2, a+1))
                subplot_title=str(img_name[a])
                axes[-1].set_title(subplot_title)  
                plt.axis('off')
                plt.imshow(Image.open(img_list[a]))
            fig.tight_layout()
            st.write(fig)
elif s_type == 'Hybrid Search':
        query_text = st.text_input('Enter some text for Hybrid search: bear,tower,city etc.')
        if query_text:
            query_vector  = searching.Hybrid_Search(query_text)
            img_list,img_name = searching.search_by_vector(query_vector)
            st.subheader(f"Similar Result Found For Keyword: {query_text}")
            # Visualize the result
            axes=[]
            fig=plt.figure(figsize=(20,20))
            for a in range(2*2):
                #score = scores[a]
                axes.append(fig.add_subplot(2, 2, a+1))
                subplot_title=str(img_name[a])
                axes[-1].set_title(subplot_title)  
                plt.axis('off')
                plt.imshow(Image.open(img_list[a]))
            fig.tight_layout()
            st.write(fig)
