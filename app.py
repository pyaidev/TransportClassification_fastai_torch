import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

st.title('Transportni klassifikatsiya qiluvchi model: Airplane, Car, Boat')

file = st.file_uploader('Rasm yuklang', type=['png', 'jpg', 'jpeg'])

if file is not None:
    st.image(file)
    img = PILImage.create(file)
    model = load_learner('transport_model.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Extimollik: {probs[pred_id]*100:.1f}%")

    flg=px.bar(x=model.dls.vocab, y=probs)
    st.plotly_chart(flg)


st.markdown('''Developed by Nurmuhammad Mashrapov''')
