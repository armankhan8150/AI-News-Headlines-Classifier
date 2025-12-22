import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import altair as alt
import numpy as np # Need numpy to generate the tick values

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="News Classifier | AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ensure this folder exists in your project directory
    model_path = "./bert-news-classifier"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        return tokenizer, model
    except OSError:
        return None, None

tokenizer, model = load_model()
labels = {0: "World 🌍", 1: "Sports ⚽", 2: "Business 💼", 3: "Sci/Tech 💻"}

# --- INIT CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers_logo_name.png", width=200)
    st.header("About")
    st.markdown("""
    **Task:** News Topic Classification
    **Model:** BERT (Fine-Tuned)
    **Dataset:** AG News
    
    This model classifies news headlines into four categories:
    - World
    - Sports
    - Business
    - Sci/Tech
    """)
    st.markdown("---")
    
    st.header("📘 User Guide")
    st.markdown("""
    **How to use:**
    Type a news headline to see how the AI classifies it.
    
    ✅ **Good Examples:**
    * *"SpaceX launches new satellite into orbit."*
    * *"Stock markets rally after interest rate cut."*
    
    ❌ **Limitations:**
    * It is **not** a chatbot (avoid "How are you?").
    * Works best with full English sentences.
    """)
    
    # Clear History Button
    if st.button("Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption("Built with BERT & Streamlit")

# --- MAIN CHAT INTERFACE ---
st.title("🤖 AI News Headlines Classifier")
st.caption("Powered by Fine-Tuned BERT-Base")
st.write("Chat with the model to classify news headlines. Scroll up to view history.")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there is a chart saved in this message, display it
        if "chart_data" in message:
            st.altair_chart(message["chart_data"], use_container_width=True)

# 2. Input Loop
if prompt := st.chat_input("Enter a news headline here..."):
    
    # Add User Input to History
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if model:
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Prediction Logic
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.softmax(outputs.logits, dim=1)[0]
                prediction = torch.argmax(probs).item()
                confidence = probs[prediction].item()

                # --- SOFT WARNING ---
                warning_msg = ""
                if confidence < 0.6:
                    warning_msg = "⚠️ *Low confidence. The model is unsure about this input.* \n\n"

                # Formatted Output
                category_name = labels[prediction]
                response_text = f"{warning_msg}**Classification:** {category_name} \n**Confidence:** {confidence:.1%}"
                
                # --- CHART VISUALIZATION ---
                chart_df = pd.DataFrame({'Category': list(labels.values()), 'Probability': probs.numpy()})
                
                # Generate tick values from 0% to 100% in 4% increments
                tick_values = np.arange(0, 1.01, 0.04)

                # 1. Base Chart with CUSTOM X-AXIS
                base = alt.Chart(chart_df).encode(
                    x=alt.X(
                        'Probability',
                        # Define axis with title 'Probability' and custom tick values/format
                        axis=alt.Axis(
                            title='Probability', 
                            format='%', 
                            values=list(tick_values)
                        )
                    ),
                    y=alt.Y('Category', sort='-x', axis=alt.Axis(title=None, tickBand='extent')),
                    tooltip=['Category', alt.Tooltip('Probability', format='.1%')]
                )

                # 2. Bars with Gradient
                bars = base.mark_bar(cornerRadiusEnd=4).encode(
                    color=alt.Color('Probability', 
                                    scale=alt.Scale(scheme='tealblues', domain=[0, 1]), 
                                    legend=None)
                )

                # 3. Text Labels
                text = base.mark_text(align='left', dx=5, color='black', font='sans-serif').encode(
                    text=alt.Text('Probability', format='.1%')
                )
                
                # 4. Final Layout
                chart = alt.layer(bars, text).properties(
                    title=alt.TitleParams(text='Confidence Scores', anchor='start', color='gray', fontSize=14),
                    height=alt.Step(40) 
                ).configure_view(
                    strokeWidth=0
                )

                # Display in Chat
                st.markdown(response_text)
                st.altair_chart(chart, use_container_width=True)
                
                # Save Assistant Response to History
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "chart_data": chart
                })
    else:
        st.error("❌ Model not found. Please check your folder structure.")