import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import os

from csv import writer
# read csv from a github repo
st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    layout="wide",
)  
add_selectbox = st.sidebar.selectbox(
    "เลือกห้องเรียน",
    ("ม.5/1", "ม.5/2", "ม.5/3", "ม.5/11")
)
with st.sidebar:
    add_radio = st.radio(
        "เลือกข้อมูลที่ต้องการแสดง",
        ("Dashboard", "คลังรูปภาพ")
    )


with st.container():
    st.title("Crossec")
    st.subheader("Dashboard")
    if add_radio == "Dashboard":
        st.dataframe(pd.read_csv('Userdata.csv')) 
    elif add_radio == "คลังรูปภาพ":
        tissue = ['พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะปฐมภูมิ','พืชเลี้ยงคู่ส่วนลำต้นปฐมระยะภูมิ','พืชเลี้ยงคู่ส่วนรากระยะปฐมภูมิ','พืชใบเลี้ยงเดี่ยวส่วนลำต้นระยะทุติยภูมิ','พืชใบเลี้ยงเดี่ยวส่วนรากระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนลำต้นปฐมระยะทุติยภูมิ','พืชเลี้ยงคู่ส่วนรากระยะทุติยภูมิ','ใบพืช C3', 'ใบพืช C4']
        with st.container():
            path = os.path.join('./predicted/')
            cpt = sum([len(files) for r, d, files in os.walk(path)])
            for i in range(10):
                st.text("")
                st.text("")
                st.write(tissue[i])
                _path = os.path.join('./predicted/',tissue[i])
                _cpt = sum([len(files) for r, d, files in os.walk(_path)])
                array = [1,1,1]
                col = st.columns(array)
                k=0
                for j in range(_cpt):
                    if k == 3:
                        k=0
                    if k%3 == 0 and k!=0:
                        col = st.columns(array)
                    col[k].image(f"{_path}/{j}.jpg",width=400)
                    k+=1




        