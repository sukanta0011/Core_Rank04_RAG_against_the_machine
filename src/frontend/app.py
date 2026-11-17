"""Streamlit frontend for the vLLM RAG API."""
import os
import time
from typing import Any, Dict

import requests
import streamlit as st

API_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000")
SOURCES_TIMEOUT = 60      # seconds
ANSWER_TIMEOUT = 600      # the first /ask loads the LLM, which is slow
MODES = {
    "Show sources only": ("/sources", SOURCES_TIMEOUT),
    "Generate answer": ("/ask", ANSWER_TIMEOUT),
}

st.set_page_config(page_title="vLLM RAG Assistant", page_icon="🤖")
st.title("🤖 vLLM RAG Assistant")
st.markdown("Ask questions about the vLLM documentation.")


def call_api(endpoint: str, timeout: int, question: str,
             k: int) -> Dict[str, Any]:
    """POST the question to the backend and return the JSON body.

    Args:
        endpoint: "/sources" or "/ask".
        timeout: Seconds to wait for the response.
        question: Question text.
        k: Number of sources to retrieve.

    Returns:
        The parsed JSON response.

    Raises:
        requests.RequestException: On connection or HTTP errors.
    """
    response = requests.post(
        f"{API_URL}{endpoint}",
        json={"question": question, "k": k},
        timeout=timeout)
    response.raise_for_status()
    return dict(response.json())


def show_sources(data: Dict[str, Any], expanded: bool) -> None:
    """Render the retrieved source locations.

    Args:
        data: API response containing `resource_locations`.
        expanded: Whether the expander starts open.
    """
    sources = data.get("resource_locations", [])
    with st.expander(f"Retrieved sources ({len(sources)})",
                     expanded=expanded):
        if not sources:
            st.write("No sources found.")
        for idx, source in enumerate(sources, start=1):
            st.markdown(
                f"**{idx}.** `{source['file_path']}` "
                f"[{source['first_character_index']}:"
                f"{source['last_character_index']}]")


mode = st.radio("What do you want?", list(MODES), horizontal=True)
question = st.text_input(
    "Enter your question:", placeholder="How do I setup vLLM?")
k_value = st.slider(
    "Number of sources to retrieve", min_value=1, max_value=10, value=3)

if mode == "Generate answer":
    st.caption("The first answer can take a minute or two while "
               "the language model loads.")

if st.button("Ask"):
    if len(question.strip()) < 3:
        st.warning("Please enter a question (at least 3 characters).")
    else:
        endpoint, timeout = MODES[mode]
        spinner_text = ("Searching..." if endpoint == "/sources"
                        else "Searching and generating answer...")
        try:
            with st.spinner(spinner_text):
                start = time.time()
                data = call_api(endpoint, timeout, question.strip(),
                                k_value)
                elapsed = time.time() - start

            if endpoint == "/ask":
                st.subheader(f"Answer ({elapsed:.2f}s)")
                st.write(data.get("answer", "(empty answer)"))
                show_sources(data, expanded=False)
            else:
                st.subheader(f"Sources ({elapsed:.2f}s)")
                show_sources(data, expanded=True)

        except requests.ConnectionError:
            st.error(f"Cannot reach the API at {API_URL}. "
                     "Is the server running?")
        except requests.Timeout:
            st.error("The API took too long to answer.")
        except requests.HTTPError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = e.response.json().get("detail", "")
                except ValueError:
                    detail = e.response.text[:300]
            st.error(f"API error {e.response.status_code if e.response is not None else ''}: {detail}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")
