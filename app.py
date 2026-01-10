import streamlit as st
from cot_monitor.cot_extractor import extract_cot
from cot_monitor.provenance import compute_risk_score

st.set_page_config(page_title="CoT Manipulation Monitor", layout="wide")

st.title("Chain-of-Thought Manipulation Monitor")

query = st.text_area(
    "User message",
    height=120,
    placeholder="Enter a user query or conversation turn...",
)

mode = st.selectbox("Mode", ["normal", "audit_aware"])
model = st.text_input("Groq model id", value="llama-3.1-8b-instant")

if st.button("Analyze") and query.strip():
    with st.spinner("Querying model and analyzing CoT..."):
        res = extract_cot(query, mode=mode, model=model)
        cot = res["cot"]
        answer = res["answer"]
        metrics = compute_risk_score(cot)

    risk = metrics["risk"]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Risk score")
        if risk < 30:
            st.success(f"Risk: {risk}/100 (green)")
        elif risk < 70:
            st.warning(f"Risk: {risk}/100 (yellow)")
        else:
            st.error(f"Risk: {risk}/100 (red)")

        st.metric("Pattern risk", f"{metrics['pattern_risk']:.2f}")
        st.metric("CoT length factor", f"{metrics['long_cot']:.2f}")
        st.metric("Compression ratio", f"{metrics['compression_ratio']:.3f}")
        st.metric("Zlib entropy risk", f"{metrics['zlib_entropy_risk']:.2f}")

        # per-pattern bar chart
        pattern_items = {
            k.replace("pat_", ""): v
            for k, v in metrics.items()
            if k.startswith("pat_")
        }
        st.subheader("Per-pattern scores")
        st.bar_chart(pattern_items)

    with col2:
        st.subheader("Chain-of-thought")
        st.write(cot)

        st.subheader("Final answer")
        st.write(answer)
