"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from this import d
from PIL import Image
from fastai.vision.widgets import *
from fastai.vision.all import *
from torchvision import models, transforms
import torch
import streamlit as st
import time
import os
import os.path
import urllib.request
from pathlib import Path
from csv import writer
import tensorflow as tf
import pathlib
import numpy as np

st.set_page_config(
    page_title="Crossec ML",
    layout="wide",
)  
@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./model.h5')
	return model
# file_exists = os.path.exists('crossec_model.pkl')

# plt = platform.system()
# if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# MODEL_URL = "https://dl.dropboxusercontent.com/s/k66f4yi8i0mlalp/crossec_model.pkl?dl=0"
# urllib.request.urlretrieve(MODEL_URL,"crossec_model.pkl")
# learn_inf = load_learner(Path()/'crossec_model.pkl',cpu=True)
def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [150, 150])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction
st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")
def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")

def page2():
    st.markdown("# Page 2 ❄️")
    st.sidebar.markdown("# Page 2 ❄️")

def page3():
    st.markdown("# Page 3 🎉")
    st.sidebar.markdown("# Page 3 🎉")

page_names_to_funcs = {
    "Main Page": main_page,
    "Page 2": page2,
    "Page 3": page3,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
model = load_model()
tissue = [
 'ใบพืช C3', 'ใบพืช C4', 'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะทุติยภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ'
]

# def predict(image, learn):
#         """Return top 5 predictions ranked by highest probability.

#         Parameters
#         ----------
#         :param image: uploaded image
#         :type image: jpg
#         :rtype: list
#         :return: top 5 predictions ranked by highest probability
#         """
#         # create a ResNet model
#         pred, pred_idx, pred_prob = learn.predict(image)

#         classes = tissue[int(pred_idx)]
        
#         return [(classes, pred_prob[pred_idx])]


image = Image.open('Logo.png')
with st.container():
    st.image(image,width=200)

    st.subheader("Let's make cross section easier!")
    st.write("")
    file_up = st.file_uploader("Upload an image", type = ["jpg","png"])
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        pred = predict_class(np.asarray(image), model)
        class_names = ['ใบพืช C3', 'ใบพืช C4', 'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะทุติยภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ']
        result = class_names[np.argmax(pred)]
        output = 'The image is a ' + result
            # print out the top 5 prediction labels with scores
        st.success(output)
    st.write("")
    with st.expander("ส่งงาน"):
        name = st.text_input('ชื่อ-นามสกุล')
        room = st.selectbox(
            'ห้อง',
            ['ม.5/1','ม.5/2','ม.5/3','ม.5/11']
        )
        num = st.text_input('เลขที่')
        tissue = st.selectbox(
            'เลือกหัวข้อปฏิบัติการ',
            ['ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนลำต้นปฐมระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนรากระยะทุติยภูมิ','ใบพืช C3', 'ใบพืช C4']
        )
        user={'Name':name,'Room':room,'Number':num}
        if st.button('Send'):
            if name != "" and num != "":
            #     noobcopycatch = glob.glob('./predicted/'+ labels[0][0]+'/'+file_up.name)
            #     if noobcopycatch != []:
            #         st.write("อย่าโกงงงง")
            #     else:
            #         count = 0
            #         path = './predicted/' + labels[0][0]
            #         isExist = os.path.exists(path)
            #         if not isExist:
            # # Create a new directory because it does not exist 
            #             os.makedirs(path)
            #         for root_dir, cur_dir, files in os.walk(r'.\predicted'):
            #             count += len(files)
            #         with open(os.path.join(path,file_up.name), "wb") as f:
            #             f.write(file_up.getbuffer())
            #         st.success(user['Name']+ " ชั้น " + user['Room']+ " เลขที่ " +user['Number']+" ส่ง "+labels[0][0]+" แล้ว")
                count = 0
                path = './predicted/'+ room + '/' + name + '/' + tissue
                isExist = os.path.exists(path)
                if not isExist:
                        os.makedirs(path)
                for root_dir, cur_dir, files in os.walk(r'.\predicted'):
                        count += len(files)
                with open(os.path.join(path,file_up.name), "wb") as f:
                        f.write(file_up.getbuffer())
                cpt = sum([len(files) for r, d, files in os.walk(path)]) -1
                os.rename(f"{path}/{file_up.name}",f'{path}/{cpt}.jpg')
                st.success(user['Name']+ " ชั้น " + user['Room']+ " เลขที่ " +user['Number']+" ส่ง "+labels[0][0]+" แล้ว")
                path=os.path.join(room+'.csv')
                with open(path, 'a',encoding="utf-8") as csvfile:
                    writer_object = writer(csvfile)
            
                # Pass the list as an argument into
                # the writerow()
                    writer_object.writerow([name,room,num,tissue])
                    csvfile.close()
            else:
               st.error("กรุณากรอกข้อมูลให้ครบถ้วน") 
    # enable users to upload images for the model to make predictions
    st.write("")
with st.container():
    cpt = sum([len(files) for r, d, files in os.walk('./predicted/')])
    st.metric(label="Image Predicted", value=cpt, delta=1)
