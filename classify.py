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
import pathlib

st.set_page_config(
    page_title="Crossec ML",
    layout="wide",
)  
with st.sidebar:
    st.image('Logo.png',width=100)
    st.image('aib.png',width=100)
    st.header("About this Project✔")
    st.write("Project นี้ได้รับการสนับสนุนจากโครงการ [AI Builder 2022](https://ai-builders.github.io/) โดยจัดทำขึ้นเพื่อเป็นตัวช่วยในการส่งเสริมการทำปฏิบัติการภาพตัดขวาง โดยทีมผู้พัฒนาได้จัดทำ Dataset และเผยแพร่ [Open Source](https://www.kaggle.com/datasets/earthwttw/plant-tissue-cross-section-dataset) ไว้บนเว็บไซต์ Kaggle",unsafe_allow_html=True)
    st.header("Developer👨‍💻")
    st.text("Wittawat Kitipatthavorn\nNawapat Jongaouyporn\nSorraat Treenuson")
file_exists = os.path.exists('crossec_model.pkl')

plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

@st.cache(allow_output_mutation=True,show_spinner=False,ttl=1800,max_entries=2,persist=True)
def load_model(): return load_learner(Path()/'crossec_model.pkl',cpu=True)

tissue = [
'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ', 'ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ', 'ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ'
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
    file_up = st.file_uploader("Upload an image", type = "jpeg,jpg")
    if file_up is not None:
        image = PILImage.create(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        labels = predict(image,load_model())

        score = labels[0][1]
            # print out the top 5 prediction labels with scores
        if st.success(f"Result: {labels[0][0]} {score*100:.02f}%"):
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
                [
'ใบพืช C3', 'ใบพืช C4', 'รากพืชใบเลี้ยงคู่ระยะปฐมภูมิ','รากพืชใบเลี้ยงคู่ระยะทุติยภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงคู่ระยะทุติยภูมิ','รากพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ','ลำต้นพืชใบเลี้ยงเดี่ยวระยะปฐมภูมิ'
]
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