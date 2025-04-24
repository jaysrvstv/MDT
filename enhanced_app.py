import streamlit as st
import openai
import pandas as pd
import os
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# File paths
memory_file = "memory.csv"
embedding_file = "embeddings.npy"

# Initialize memory file
if not os.path.exists(memory_file):
    df = pd.DataFrame(columns=["Type", "Content"])
    df.to_csv(memory_file, index=False)

# App UI
st.set_page_config(page_title="MDT (My Digital Twin)", layout="centered")
st.title("🧠 MDT (My Digital Twin) Assistant")
mode = st.sidebar.radio("Choose mode", ["🎙️ Voice Input", "🗒️ Task Entry", "🔍 Memory Recall", "🤖 Chat Assistant"])

# Load memory
df = pd.read_csv(memory_file)
memory = df["Content"].tolist()

# Utility: Generate embedding
def get_embedding(text):
    return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

# Voice Input Mode
if mode == "🎙️ Voice Input":
    st.header("🎙️ Upload a Voice Note")
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a"])
    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name
        transcript = openai.Audio.transcribe("whisper-1", open(tmp_path, "rb"))
        st.success("Transcribed Text:")
        st.write(transcript["text"])

        # Classify
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify this into one: Work, Personal, Finance, Research"},
                {"role": "user", "content": transcript["text"]},
            ],
        )
        task_type = response.choices[0].message.content.strip()
        df = pd.concat([df, pd.DataFrame([{"Type": task_type, "Content": transcript["text"]}])], ignore_index=True)
        df.to_csv(memory_file, index=False)
        st.success(f"Stored as **{task_type}** task.")

# Task Entry Mode
elif mode == "🗒️ Task Entry":
    st.header("🗒️ Enter a Task or Thought")
    task_input = st.text_area("What do you want to record?")
    if st.button("Store Thought"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify this into one: Work, Personal, Finance, Research"},
                {"role": "user", "content": task_input},
            ],
        )
        task_type = response.choices[0].message.content.strip()
        df = pd.concat([df, pd.DataFrame([{"Type": task_type, "Content": task_input}])], ignore_index=True)
        df.to_csv(memory_file, index=False)
        st.success(f"Stored as **{task_type}** task.")

# Memory Recall Mode
elif mode == "🔍 Memory Recall":
    st.header("🔍 Ask Me Anything You've Said Before")
    query = st.text_input("Your query:")
    if st.button("Recall Memory"):
        if memory:
            query_vec = np.array(get_embedding(query)).reshape(1, -1)
            memory_vecs = np.array([get_embedding(m) for m in memory])
            scores = cosine_similarity(query_vec, memory_vecs).flatten()
            top_index = np.argmax(scores)
            st.info(f"🧠 Closest Memory Match:

{memory[top_index]}")
        else:
            st.warning("No memories stored yet.")

# Chat Assistant Mode
elif mode == "🤖 Chat Assistant":
    st.header("🤖 Chat with MDT")
    prompt = st.text_input("Ask something:")
    if st.button("Get Response"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        st.write(response.choices[0].message.content)
