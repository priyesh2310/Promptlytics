import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os
import requests
import re
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Page setup
st.set_page_config(page_title="Promptlytics", layout="centered")
st.title("📊 Promptlytics (Ask Anything from CSV)")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your CSV file", type="csv")

# Initialize session state
if "question" not in st.session_state:
    st.session_state["question"] = ""

# Load and preview CSV
df = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)

        # Limit for large files
        MAX_ROWS = 20000
        if df.shape[0] > MAX_ROWS:
            st.warning(f"⚠️ File has {df.shape[0]} rows. Only using first {MAX_ROWS} rows for performance.")
            df = df.head(MAX_ROWS)

        st.success("✅ CSV uploaded successfully!")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")

# Suggested questions
if df is not None:
    suggested_questions = []
    cols = df.columns.tolist()
    for i in range(min(3, len(cols))):
        for j in range(i + 1, min(i + 4, len(cols))):
            suggested_questions += [
                f"Average {cols[i]} by {cols[j]}",
                f"Top 5 {cols[j]} by {cols[i]}",
                f"Distribution of {cols[i]}"
            ]
    suggested_questions = list(set(suggested_questions))[:5]

    if suggested_questions:
        st.markdown("💡 **Try asking:**")
        for q in suggested_questions:
            if st.button(q):
                st.session_state["question"] = q

    # Prompt input
    prompt = st.text_input("💬 Ask anything:", value=st.session_state["question"])

    # Handle submit
    if st.button("Submit") and prompt:
        st.info("🧠 Generating answer using **Mistral 7B (Free)**...")

        try:
            # Sample data for LLM
            csv_preview = df.sample(20, random_state=42).to_csv(index=False)

            full_prompt = f"""You are a helpful data analyst. Here's a sample of a CSV:

{csv_preview}

Answer this user query: {prompt}

Be specific. Include values or analysis if necessary.
Do NOT include Python code in your response.
Respond in plain English or table format only.
"""

            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "https://your-app-name.com",
                "X-Title": "Promptlytics",
                "Content-Type": "application/json",
            }

            payload = {
                "model": "mistralai/mistral-7b-instruct:free",
                "messages": [{"role": "user", "content": full_prompt}],
            }

            response = requests.post("https://openrouter.ai/api/v1/chat/completions",
                                     headers=headers, json=payload)

            if response.status_code == 200:
                reply = response.json()["choices"][0]["message"]["content"]

                # Clean and parse LLM response
                def clean_response(text):
                    text = re.sub(r"```(?:python)?(.*?)```", "", text, flags=re.DOTALL)
                    lines = text.strip().split('\n')
                    table_lines = [line for line in lines if ',' in line or '\t' in line]
                    if table_lines and len(table_lines) > 1:
                        try:
                            preview_df = pd.read_csv(io.StringIO('\n'.join(table_lines)))
                            return "📝 Response (Cleaned):", preview_df
                        except:
                            pass
                    return "📝 Response (Cleaned):", text.strip()

                title, cleaned = clean_response(reply)
                st.success(title)
                if isinstance(cleaned, pd.DataFrame):
                    st.dataframe(cleaned)
                else:
                    st.markdown(cleaned)

            else:
                st.error(f"❌ Error {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

        # Chart Generation from Prompt
        st.markdown("---")
        st.markdown("📈 **Chart based on your question**")

        mentioned_cols = [col for col in df.columns if col.lower() in prompt.lower()]
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        x_col = y_col = None
        if len(mentioned_cols) >= 2:
            x_col, y_col = mentioned_cols[1], mentioned_cols[0]
        elif len(mentioned_cols) == 1 and mentioned_cols[0] in numeric_cols:
            y_col = mentioned_cols[0]
            other_cols = [col for col in df.columns if col != y_col]
            x_col = other_cols[0] if other_cols else y_col
        elif len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[:2]

        if x_col and y_col:
            try:
                df_sorted = df.sort_values(by=y_col, ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=df_sorted, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"{y_col} by {x_col} (Top 10)")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"⚠️ Could not generate chart: {e}")
        else:
            st.warning("⚠️ Could not determine suitable columns for chart from your question.")

