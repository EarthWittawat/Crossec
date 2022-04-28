"""Create an Image Classification Web App using PyTorch and Streamlit."""
# import libraries
from PIL import Image
from torchvision import models, transforms
import torch
import streamlit as st
import time
import os
import glob

# set title of app
def predict(image):
        """Return top 5 predictions ranked by highest probability.

        Parameters
        ----------
        :param image: uploaded image
        :type image: jpg
        :rtype: list
        :return: top 5 predictions ranked by highest probability
        """
        # create a ResNet model
        resnet = models.resnet101(pretrained = True)

        # transform the input image through resizing, normalization
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
                )])

        # load the image, pre-process it, and make predictions
        img = Image.open(image)
        batch_t = torch.unsqueeze(transform(img), 0)
        resnet.eval()
        out = resnet(batch_t)

        with open('imagenet_classes.txt') as f:
            classes = [line.strip() for line in f.readlines()]

        # return the top 5 predictions ranked by highest probabilities
        prob = torch.nn.functional.softmax(out, dim = 1)[0] * 100
        _, indices = torch.sort(out, descending = True)
        # predict_progress = st.progress(0)
        # for percent_complete in range(100):
        #     time.sleep(0.05)
        #     predict_progress.progress(percent_complete + 1)
        return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]


image = Image.open('Logo.png')
with st.container():
    st.image(image,width=200)

    st.subheader("Let's make cross section easier!")
    st.write("")
    file_up = st.file_uploader("Upload an image", type = "jpg")
    if file_up is not None:
        # display image that user uploaded
        image = Image.open(file_up)
        st.image(image, caption = 'Uploaded Image.', use_column_width = True)
        st.write("")
        labels = predict(file_up)
        score = int(labels[0][1])
        score = str(score)
            # print out the top 5 prediction labels with scores
        st.success("Result: "+ labels[0][0] + " " + score+"%")
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
            ['พืชใบเลี้ยงเดี่ยวส่วนลำต้น','พืชใบเลี้ยงเดี่ยวส่วนราก','พืชเลี้ยงคู่ส่วนลำต้น','พืชเลี้ยงคู่ส่วนลำต้น','ใบพืช C3', 'ใบพืช C4']
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
                st.success(user['Name']+ " ชั้น " + user['Room']+ " เลขที่ " +user['Number']+" ส่ง "+labels[0][0]+" แล้ว")
            else:
               st.error("กรุณากรอกข้อมูลให้ครบถ้วน") 
    # enable users to upload images for the model to make predictions
    st.write("")
with st.container():
    cpt = sum([len(files) for r, d, files in os.walk('./predicted/')])
    st.metric(label="Image Predicted", value=cpt, delta=1)
