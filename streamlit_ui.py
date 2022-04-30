"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
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
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

if os.path.exists('crossec_model.pkl') == True:
    MODEL_URL = "https://dl.dropboxusercontent.com/s/9ayzkc9jcam5adf/crossec_model.pkl?dl=0"
    urllib.request.urlretrieve(MODEL_URL,"crossec_model.pkl")
learn_inf = load_learner(Path()/'crossec_model.pkl')

tissue = [
 'C3DicotStemSecondary', 'C3DicotStemPrimary'
]

def predict(image, learn):
        """Return top 5 predictions ranked by highest probability.

        Parameters
        ----------
        :param image: uploaded image
        :type image: jpg
        :rtype: list
        :return: top 5 predictions ranked by highest probability
        """
        # create a ResNet model
        pred, pred_idx, pred_prob = learn.predict(image)

        classes = tissue[int(pred_idx)]
        
        return [(classes, pred_prob[pred_idx])]


image = Image.open('Logo.png')
with st.container():
    st.image(image,width=200)

    st.subheader("Let's make cross section easier!")
    st.write("")
    file_up = st.file_uploader("Upload an image", type = "jpg")
    if file_up is not None:
        # display image that user uploaded
        image = PILImage.create(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        labels = predict(image,learn_inf)
        score = labels[0][1]
            # print out the top 5 prediction labels with scores
        st.success(f"Result: {labels[0][0]} {score*100:.02f}%")
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
            ['พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะปฐมภูมิ','พืชเลี้ยงคู่ส่วนลำต้นปฐมระยะภูมิ','พืชเลี้ยงคู่ส่วนรากระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะทุติยภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนลำต้นปฐมระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนรากระยะทุติยภูมิ','ใบพืช C3', 'ใบพืช C4']
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
                path = './predicted/' + tissue
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
                path=os.path.join('Userdata.csv')
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
