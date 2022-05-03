import time  # to simulate a real time data, time loop

from PIL import Image
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import os

from csv import writer
# read csv from a github repo
st.set_page_config(
    page_title="Crossec Dashboard",
    layout="wide",
)  
room = st.sidebar.selectbox(
    "เลือกห้องเรียน",
    ("ม.5/1", "ม.5/2", "ม.5/3", "ม.5/11")
)
with st.sidebar:
    add_radio = st.radio(
        "เลือกข้อมูลที่ต้องการแสดง",
        ("Dashboard", "คลังรูปภาพ")
    )

image = Image.open('Logo.png')
with st.container():
    st.image(image,width=200)
    st.subheader("Let's make cross section easier!")
    st.write("")
    st.subheader("Dashboard")
    if add_radio == "Dashboard":
        st.dataframe(pd.read_csv(room+'.csv')) 
    elif add_radio == "คลังรูปภาพ":
        tissue = ['พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะปฐมภูมิ','พืชใบเลี้ยงคู่ส่วนลำต้นระยะปฐมภูมิ','พืชใบเลี้ยงคู่ส่วนรากระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะทุติยภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะทุติยภูมิ','พืชใบเลี้ยงคู่ส่วนลำต้นปฐมระยะทุติยภูมิ','พืชใบเลี้ยงคู่ส่วนรากระยะทุติยภูมิ','ใบพืช C3', 'ใบพืช C4']
        with st.container():
            path = './predicted/'+ room
            path = os.path.join(path)
            cpt = sum([len(files) for r, d, files in os.walk(path)])
            student = []
            stu_num = len(os.listdir(path))
            tissue_available = []
            for stu in range(stu_num):
                student.append(os.listdir(path)[stu])
                for j in range(10):
                    if os.path.isdir(path + '/' + student[stu] + '/' + tissue[j]):
                        tissue_available.append(tissue[j])
            for i in range(len(set(tissue_available))):
                    st.text("")
                    st.write(tissue_available[i])
                    for stu in range(stu_num):
                        _path = os.path.join(path,student[stu],tissue_available[i])
                        _cpt = sum([len(files) for r, d, files in os.walk(_path)])
                        array = [1,1,1]
                        col = st.columns(array)
                        k=0
                        for j in range(_cpt):
                            if k == 3:
                                k=0
                            if k%3 == 0 and k!=0:
                                col = st.columns(array)
                            col[k].image(f"{_path}/{j}.jpg",width=400, caption=student[stu])
                            k+=1




        