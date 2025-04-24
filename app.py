import streamlit as st
import openai
import pandas as pd
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# App Title
st.title("MDT (My Digital Twin) 🔮")

# Sidebar
st.sidebar.header("Navigation")
mode = st.sidebar.radio("Choose mode:", ["Task Entry", "Memory Recall", "Chat Assistant"])

# Simple CSV-based memory storage
memory_file = "memory.csv"

# Initialize CSV if not present
if not os.path.exists(memory_file):
    df = pd.DataFrame(columns=["Type", "Content"])
    df.to_csv(memory_file, index=False)

# Task Entry Mode
if mode == "Task Entry":
    st.header("🗒️ Enter a new task or thought:")
    task_input = st.text_area("Your task/thought:")
    if st.button("Classify & Store"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Classify into one: Work, Personal, Finance, Research"},
                {"role": "user", "content": task_input},
            ],
        )
        task_type = response.choices[0].message.content.strip()
        df = pd.read_csv(memory_file)
        df = pd.concat([df, pd.DataFrame([{"Type": task_type, "Content": task_input}])], ignore_index=True)
        df.to_csv(memory_file, index=False)
        st.success(f"Stored under category: **{task_type}**")

# Memory Recall Mode
elif mode == "Memory Recall":
    st.header("🔍 Recall your past notes/tasks:")
    query = st.text_input("What do you want to recall?")
    if st.button("Search Memory"):
        df = pd.read_csv(memory_file)
        matches = df[df["Content"].str.contains(query, case=False)]
        if not matches.empty:
            st.table(matches)
        else:
            st.warning("No matching memory found.")

# Chat Assistant Mode
elif mode == "Chat Assistant":
    st.header("🤖 Chat with your MDT Assistant:")
    prompt = st.text_input("Ask me anything:")
    if st.button("Send"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content
        st.write(answer)
