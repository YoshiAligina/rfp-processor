import os
import streamlit as st
from ocr_utils import is_scanned, convert_scanned, extract_text
from model_utils import predict_document_probability
from data_utils import add_entry_to_db, load_db

PDF_FOLDER = "rfps"

def main():
    st.title("RFP Worthiness Predictor & Batch Uploader (CSV Version)")

    # --- Multiple Upload Section ---
    st.header("Batch Upload RFPs for Training")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        os.makedirs(PDF_FOLDER, exist_ok=True)
        entries_to_add = []

        # Let user enter metadata for each file
        with st.form("batch_metadata_form"):
            st.write("Enter information for each uploaded file:")
            titles = []
            senders = []
            decisions = []

            for uploaded_file in uploaded_files:
                st.markdown(f"#### {uploaded_file.name}")
                title = st.text_input(
                    f"Title for {uploaded_file.name}",
                    value=os.path.splitext(uploaded_file.name)[0],
                    key=f"title_{uploaded_file.name}"
                )
                sender = st.text_input(
                    f"From (Sender/Company) for {uploaded_file.name}",
                    key=f"sender_{uploaded_file.name}"
                )
                decision = st.selectbox(
                    f"Final Decision for {uploaded_file.name}",
                    ["Pending", "Approved", "Denied"],
                    key=f"decision_{uploaded_file.name}"
                )
                titles.append(title)
                senders.append(sender)
                decisions.append(decision)

            submitted = st.form_submit_button("Process and Add to Database")
        
        if submitted:
            for idx, uploaded_file in enumerate(uploaded_files):
                pdf_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                # OCR and text extraction
                processed_path = pdf_path
                if is_scanned(pdf_path):
                    st.info(f"Applying OCR to scanned PDF: {uploaded_file.name}")
                    processed_path = convert_scanned(pdf_path, pdf_path.replace(".pdf", "_ocr.pdf"))
                text = extract_text(processed_path)
                excerpt = text[:500].replace('\n', ' ').replace('\r', ' ') if text else ""
                prob = predict_document_probability(text)
                entry = {
                    "filename": uploaded_file.name,
                    "title": titles[idx],
                    "sender": senders[idx],
                    "decision": decisions[idx],
                    "probability": prob,
                    "excerpt": excerpt
                }
                add_entry_to_db(entry)
                st.success(f"{uploaded_file.name} added to database!")
    
    # --- Review Database Section ---
    st.header("RFP Database")
    df = load_db()
    if not df.empty:
        df_sorted = df.sort_values("probability", ascending=False)
        for _, row in df_sorted.iterrows():
            with st.expander(f"{row['title']} ({row['filename']}) — {row['probability']:.2%}"):
                st.write(f"**From:** {row['sender']}")
                st.write(f"**Decision:** {row['decision']}")
                st.write(f"**Probability of being 'worth it':** {row['probability']:.2%}")
                st.write(f"**Excerpt:** {row['excerpt']}")
        st.download_button("Download CSV", data=df_sorted.to_csv(index=False),
                           file_name="rfp_db.csv", mime="text/csv")
    else:
        st.write("No RFPs in the database yet.")

if __name__ == "__main__":
    main()
