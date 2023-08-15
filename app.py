import streamlit as st
from fastai.vision.all import *
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

        pred_class, _, probs = model.predict(img)
        pred_class = str(pred_class)
        pred_prob = probs[model.dls.vocab.o2i[pred_class]] * 100

        if pred_prob >= 90:
            st.success(f"Bashorat: {pred_class}")
            st.info("Extimollik: {:.1f}%".format(pred_prob))

            flg = px.bar(x=model.dls.vocab, y=probs)
            st.plotly_chart(flg)

            st.image(img.to_thumb(300, 300), caption=f'Natija: {pred_class}', use_column_width=True)
        else:
            st.error("Noto'g'ri kategoriya yoki noto'g'ri ehtimollik. Iltimos, Car, Airplane yoki Boat turlariga tegishli va ehtimollik 90% dan yuqori rasmlarni yuklang.")

st.markdown('''Developed by Nurmuhammad Mashrapov 👨🏻‍💻''')
