import os
import shutil
import streamlit as st
import pandas as pd
from ocr_utils import is_scanned, convert_scanned, extract_text
from model_utils import predict_document_probability
from data_utils import add_entry_to_db, load_db, update_decision 


PDF_FOLDER = "rfps"

def main():
    st.title("📄 RFP Predictor & Batch Uploader (Longformer Version)")

    st.header("Batch Upload RFPs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
# Handle batch metadata input and processing 
    if uploaded_files:
        os.makedirs(PDF_FOLDER, exist_ok=True)
        with st.form("batch_metadata_form"):
            st.write("Enter metadata for each uploaded PDF:")
            titles, senders, decisions = [], [], []
            for uploaded_file in uploaded_files:
                st.markdown(f"#### 📎 `{uploaded_file.name}`")
                titles.append(
                    st.text_input(
                        f"Title for {uploaded_file.name}",
                        value=os.path.splitext(uploaded_file.name)[0],
                        key=f"title_{uploaded_file.name}"
                    )
                )
                senders.append(
                    st.text_input(
                        f"From (Sender/Company)", key=f"sender_{uploaded_file.name}"
                    )
                )
                # Implement logic to extract and display key information from the PDF
                decisions.append(
                    st.selectbox(
                        "Final Decision",
                        ["Pending", "Approved", "Denied"],
                        key=f"decision_{uploaded_file.name}"
                    )
                )
            submitted = st.form_submit_button(" Process and Add All")
# Process each uploaded PDF 
        if submitted:
            for idx, uploaded_file in enumerate(uploaded_files):
                # Save uploaded PDF
                pdf_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Handle OCR if needed but skipping for now as neither are installed. 
                processed_path = pdf_path
                if is_scanned(pdf_path):
                    if shutil.which("tesseract") and shutil.which("gs"):
                        st.info(f" Applying OCR to scanned PDF: {uploaded_file.name}")
                        processed_path = convert_scanned(
                            pdf_path,
                            pdf_path.replace(".pdf", "_ocr.pdf")
                        )
                    else:
                        st.warning("OCR skipped. Install Tesseract + Ghostscript to enable it.")

                try:
                    text = extract_text(processed_path)
                    excerpt = text[:500].replace('\n', ' ').replace('\r', ' ') if text else ""
                    prob = predict_document_probability(text)
                except Exception as e:
                    st.error(f"Error processing `{uploaded_file.name}`: {str(e)}")
                    continue

                entry = {
                    "filename": uploaded_file.name,
                    "title": titles[idx],
                    "sender": senders[idx],
                    "decision": decisions[idx],
                    "probability": prob,
                    "excerpt": excerpt
                }

                add_entry_to_db(entry)
                st.success(f" {uploaded_file.name} processed and added!")

    st.header(" RFP Database Review")
    df = load_db()
    if not df.empty:
        df_sorted = df.sort_values("probability", ascending=False)

        # Download CSV
        st.download_button(
            "⬇ Download CSV",
            data=df_sorted.to_csv(index=False),
            file_name="rfp_db.csv",
            mime="text/csv"
        )

        # PDF Downloads and Editable Decision
        for _, row in df_sorted.iterrows():
            with st.expander(f"{row['title']} ({row['filename']}) — {row['probability']:.2%}"):
                st.write(f"**From:** {row['sender']}")
                st.write(f"**Probability of being 'worth it':** {row['probability']:.2%}")
                st.write(f"**Excerpt:** {row['excerpt']}")

                # Editable decision
                current_decision = row['decision']
                new_decision = st.selectbox(
                    f"Update decision for {row['filename']}",
                    options=["Pending", "Approved", "Denied"],
                    index=["Pending", "Approved", "Denied"].index(current_decision),
                    key=f"decision_update_{row['filename']}"
                )

                if new_decision != current_decision:
                    if st.button(f" Save New Decision for {row['filename']}", key=f"save_decision_{row['filename']}"):
                        update_decision(row['filename'], new_decision)
                        st.success(f"Updated decision to '{new_decision}'")
                        st.rerun()  # Refresh after update

                # PDF download
                file_path = os.path.join(PDF_FOLDER, row['filename'])
                if os.path.exists(file_path):
                    with open(file_path, "rb") as f:
                        st.download_button(
                            f" Download {row['filename']}",
                            data=f.read(),
                            file_name=row['filename'],
                            mime="application/pdf"
                        )
    else:
        st.write("ℹ️ No RFPs in the database yet.")
if __name__ == "__main__":
    main()
