import codecs
import streamlit as st
import requests

st.title("🤖 Amptudix RAG Chat")

# Session state for chat history in the UI
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("📁 Knowledge Base")
    upload_file = st.file_uploader(
        "upload a PDF or TXT", type=["PDF", "txt"], key="file_uploader"
    )

    if upload_file is not None:
        if st.session_state.get("last_uploaded_file") != upload_file.name:

            with st.status("Reading and embedding...", expanded=True) as status:
                mime_type = (
                    "application/pdf"
                    if upload_file.name.endswith("pdf")
                    else "text/plain"
                )

                files = {
                    "file": (
                        upload_file.name,
                        upload_file.getvalue(),
                        mime_type,
                    )
                }
                response = requests.post(
                    "http://localhost:8000/file/upload", files=files
                )

                if response.status_code == 200:
                    st.session_state["last_uploaded_file"] = upload_file.name
                    st.toast("Brain updated successfully!")
                    status.update(
                        label="Ingestion complete!", state="complete", expanded=False
                    )
                else:
                    st.error("Failed to upload document. ")

            st.success(f"File '{upload_file.name}' is ready!")
        else:
            st.success(f"File '{upload_file.name}' is already indexed!")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about Ashan..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call your FastAPI backend
    with st.spinner("Thinking..."):
        response = requests.post(
            "http://localhost:8000/chat", json={"query": prompt}, stream=True
        )

        sources = response.headers.get("X-Sources", "")

    # Display assistant response
    with st.chat_message("assistant"):

        def stream_generator():
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            for chunk in response.iter_content(chunk_size=None):
                yield decoder.decode(chunk, final=False)

        full_answer = st.write_stream(stream_generator())

        if sources:
            st.caption(f"Sources: {sources}")

    st.session_state.messages.append({"role": "assistant", "content": full_answer})
