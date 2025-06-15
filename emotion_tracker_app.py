# Moodly - אפליקציה לניתוח רגשות מווידאו 📱

import streamlit as st
import cv2
from fer import FER
import tempfile
import os
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Moodly", layout="centered")
st.title("📱 Moodly: ניתוח רגשות מהפנים")

st.markdown("""
העלה סרטון קצר (1-2 שניות) ואנו נזהה עבורך את הרגש המרכזי בכל פריים ונפיק ממוצע רגשי יומי 🧠📊
""")

uploaded_file = st.file_uploader("העלה סרטון MP4", type=["mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.video(tfile.name)
    st.success("✅ הסרטון נטען בהצלחה. מתחילים בניתוח רגשי...")

    cap = cv2.VideoCapture(tfile.name)
    detector = FER(mtcnn=True)

    emotions_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:  # נחלץ כל פריים חמישי
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            emotion, score = detector.top_emotion(rgb_frame)
            if emotion:
                emotions_list.append((emotion, score))

        frame_count += 1

    cap.release()
    os.remove(tfile.name)

    if emotions_list:
        st.subheader("📊 ממוצע רגשי מתוך הפריימים:")
        emo_dict = {}
        for emo, score in emotions_list:
            if emo in emo_dict:
                emo_dict[emo].append(score)
            else:
                emo_dict[emo] = [score]

        avg_emotions = {k: np.mean(v) for k, v in emo_dict.items()}

        fig, ax = plt.subplots()
        ax.bar(avg_emotions.keys(), avg_emotions.values())
        ax.set_ylabel("עוצמת רגש")
        ax.set_title("ממוצע רגשות מהסרטון")
        st.pyplot(fig)

        top_emo = max(avg_emotions, key=avg_emotions.get)
        st.success(f"🎉 הרגש המרכזי שזוהה: **{top_emo}**")
    else:
        st.warning("😕 לא זוהו רגשות בסרטון. נסו סרטון ברור יותר עם פנים קרובות.")
