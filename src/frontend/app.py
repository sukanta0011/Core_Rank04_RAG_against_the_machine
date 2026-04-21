import streamlit as st
import requests
import time

# 1. Page Configuration
st.set_page_config(page_title="vLLM RAG Assistant", page_icon="🤖")

st.title("🤖 vLLM RAG Assistant")
st.markdown("Ask questions about the vLLM documentation.")

# 2. User Input
question = st.text_input("Enter your question:", placeholder="How do I setup vLLM?")
k_value = st.slider("Number of sources to retrieve", min_value=1, max_value=10, value=3)

# 3. The Logic
if st.button("Ask AI"):
    if question:
        with st.spinner("Searching and generating answer..."):
            try:
                start = time.time()
                # We talk to our FastAPI container
                response = requests.post(
                    "http://127.0.0.1:8000/ask",
                    json={"question": question, "k": k_value}
                )
                response.raise_for_status()
                data = response.json()

                # Display Results

                st.subheader(f"Answer ({(time.time() - start):.03f}s)")
                st.write(data["answer"])

                with st.expander("View Retrieved Sources"):
                    for idx, source in enumerate(data["resource_locations"]):
                        st.write(f"**Source {idx+1}:** `{source['file_path']}`")
            
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
    else:
        st.warning("Please enter a question first.")