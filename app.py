import streamlit as st
import pandas as pd
import os
from src.parser import build_dataset
from src.rag_pipeline import RAGIndex
from src.groq_client import generate_answer

st.set_page_config(page_title="Hasamex Expert Call Analyst", layout="wide")
st.title("Expert Call Intelligence Assistant")

@st.cache_resource
def load_index():
    if not os.path.exists("data/processed_chunks.csv"):
        df = build_dataset()
        df.to_csv("data/processed_chunks.csv", index=False)
    else:
        df = pd.read_csv("data/processed_chunks.csv")
    return RAGIndex(df), df

rag, full_df = load_index()

tab1, tab2, tab3 = st.tabs(["Interview Guide", "Cross-Call Themes", "Ask Transcripts"])

GUIDE_QUESTIONS = [
    "How would you describe current adoption of robotic surgery in your market?",
    "What are the main barriers to adoption?",
    "How important are hospital budgets and ROI in purchasing decisions?",
    "How important are surgeon training and clinical outcomes?",
    "What adoption trend do you expect over the next 3-5 years?",
    "What is the typical hospital decision-making timeline for purchasing a new robotic system?"
]

with tab1:
    selected_q = st.selectbox("Select interview guide question", GUIDE_QUESTIONS)
    if st.button("Generate Answer", key="guide_btn"):
        evidence = rag.search(selected_q, top_k=6)
        answer = generate_answer(selected_q, evidence)
        st.markdown(answer)
        st.subheader("Supporting Evidence")
        for _, row in evidence.iterrows():
            st.markdown(f"**{row['country']} — {row['expert_name']} ({row['timestamp']})**")
            st.markdown(f"> {row['answer']}")
            st.caption(row["source_file"])

with tab2:
    st.subheader("Cross-Call Themes and Disagreements")
    st.caption(
        "This analysis uses all timestamped expert responses from "
        "France, Germany, and the United Kingdom."
    )

    if st.button("Generate Theme Analysis", key="themes_btn"):
        theme_query = """
        Compare all three expert calls across France, Germany, and the United Kingdom.

        Return these sections:
        1. Common themes: points supported by two or more experts.
        2. Disagreements or country-specific differences: clearly name each country.
        3. Uncertainty or conditions: identify statements that depend on funding,
           training capacity, utilisation, or budget cycles.
        4. Key takeaway: a concise evidence-based conclusion.

        Use only the provided evidence. Cite every point using this exact format:
        [Country | Expert Name | MM:SS].
        Do not invent facts or citations.
        """

        with st.spinner("Comparing all expert calls..."):
            answer = generate_answer(theme_query, full_df)

        st.markdown(answer)

        with st.expander("View all evidence reviewed"):
            for _, row in full_df.iterrows():
                st.markdown(
                    f"**{row['country']} — {row['expert_name']} "
                    f"({row['timestamp']})**"
                )
                st.markdown(f"> {row['answer']}")
                st.caption(row["source_file"])

with tab3:
    st.subheader("Ask Across Transcripts")

    country_filter = st.selectbox(
        "Choose transcript scope",
        ["All countries", "France", "Germany", "United Kingdom"],
        key="country_filter"
    )

    user_q = st.text_input(
        "Ask a question",
        placeholder="Example: What are the barriers to robotic surgery adoption in France?"
    )

    if st.button("Ask", key="ask_btn") and user_q:
        search_query = user_q

        if country_filter != "All countries":
            filtered_df = full_df[full_df["country"] == country_filter].reset_index(drop=True)
            filtered_rag = RAGIndex(filtered_df)
            evidence = filtered_rag.search(search_query, top_k=min(6, len(filtered_df)))
        else:
            evidence = rag.search(search_query, top_k=6)

        with st.spinner("Finding evidence..."):
            answer = generate_answer(user_q, evidence)

        st.markdown(answer)
        st.subheader("Supporting Evidence")

        for _, row in evidence.iterrows():
            st.markdown(
                f"**{row['country']} — {row['expert_name']} "
                f"({row['timestamp']})**"
            )
            st.markdown(f"> {row['answer']}")
            st.caption(row["source_file"])