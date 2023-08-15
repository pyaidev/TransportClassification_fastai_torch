import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

st.title('Transportni klassifikatsiya qiluvchi model: Samalyot, Mashina yoki Qayiq')

file = st.file_uploader('Rasm yuklang', type=['png', 'jpg', 'jpeg'])

max_image_size = 5 * 1024 * 1024
expected_categories = ["Car", "Airplane", "Boat"]

if file is not None:
    if file.size > max_image_size:
        st.warning('Fayl hajmi juda katta. Iltimos, kichikroq fayl tanlang.')
    elif not file.type.startswith('image'):
        st.warning('Fayl turi noto\'g\'ri. Iltimos, faqat rasmlarni yuklang.')
    else:
        st.image(file)
        img = PILImage.create(file)
        model = load_learner('transport_model.pkl')

        pred, pred_id, probs = model.predict(img)

        if model.dls.vocab[pred_id] in expected_categories:
            st.success(f"Bashorat: {pred}")
            st.info(f"Extimollik: {probs[pred_id] * 100:.1f}%")

            flg = px.bar(x=model.dls.vocab, y=probs)
            st.plotly_chart(flg)

            st.image(img.to_thumb(300, 300), caption=f'Natija: {pred}', use_column_width=True)
        else:
            st.error("Noto'g'ri kategoriya. Iltimos, Car, Airplane yoki Boat turlariga tegishli rasmlarni yuklang.")


st.markdown('''Developed by Nurmuhammad Mashrapov 👨🏻‍💻‍''')
