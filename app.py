import os, time
import streamlit as st
import pandas as pd
from groq import Groq
from openai import OpenAI

st.set_page_config(page_title="LLM Comparison Dashboard", page_icon="🤖", layout="wide")
st.title("🤖 LLM Comparison Dashboard")
st.caption("Gemini 2.0 Flash · Llama 3.3 70B · DeepSeek V3 — all free APIs")
st.divider()

with st.sidebar:
    st.header("🔑 API Keys")
    st.caption("Keys stay in your session only.")
    openrouter_key = st.text_input("OpenRouter Key (for Gemini + DeepSeek)", type="password",
                                   value=os.getenv("OPENROUTER_API_KEY", ""))
    groq_key       = st.text_input("Groq Key (for Llama)", type="password",
                                   value=os.getenv("GROQ_API_KEY", ""))
    st.divider()
    st.markdown("""
    **Get free keys:**
    - [openrouter.ai](https://openrouter.ai) → covers Gemini + DeepSeek
    - [console.groq.com](https://console.groq.com) → covers Llama
    """)

# ── Gemini 2.0 Flash via OpenRouter (FREE) ──────────────────────────
def call_gemini(prompt, system_prompt, key):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp:free",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": prompt}],
            temperature=0, max_tokens=512
        )
        latency = round((time.time() - start) * 1000)
        return {
            "model"        : "Gemini 2.0 Flash",
            "response"     : resp.choices[0].message.content,
            "input_tokens" : resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "latency_ms"   : latency,
            "error"        : None
        }
    except Exception as e:
        return {"model": "Gemini 2.0 Flash", "response": None,
                "input_tokens": 0, "output_tokens": 0,
                "latency_ms": round((time.time()-start)*1000), "error": str(e)}

# ── Llama 3.3 70B via Groq (FREE) ───────────────────────────────────
def call_llama(prompt, system_prompt, key):
    client = Groq(api_key=key)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": prompt}],
            temperature=0, max_tokens=512
        )
        latency = round((time.time() - start) * 1000)
        return {
            "model"        : "Llama 3.3 70B",
            "response"     : resp.choices[0].message.content,
            "input_tokens" : resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "latency_ms"   : latency,
            "error"        : None
        }
    except Exception as e:
        return {"model": "Llama 3.3 70B", "response": None,
                "input_tokens": 0, "output_tokens": 0,
                "latency_ms": round((time.time()-start)*1000), "error": str(e)}

# ── DeepSeek V3 via OpenRouter (FREE) ───────────────────────────────
def call_deepseek(prompt, system_prompt, key):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": prompt}],
            temperature=0, max_tokens=512
        )
        latency = round((time.time() - start) * 1000)
        return {
            "model"        : "DeepSeek V3",
            "response"     : resp.choices[0].message.content,
            "input_tokens" : resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "latency_ms"   : latency,
            "error"        : None
        }
    except Exception as e:
        return {"model": "DeepSeek V3", "response": None,
                "input_tokens": 0, "output_tokens": 0,
                "latency_ms": round((time.time()-start)*1000), "error": str(e)}

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Custom Prompt", "📊 Benchmark (10 Prompts)"])

with tab1:
    st.subheader("Type any prompt — compare all 3 models instantly")
    system_prompt = st.text_area("System Prompt (optional)",
                                 value="You are a helpful assistant. Be concise.", height=80)
    user_prompt   = st.text_area("Your Prompt",
                                 placeholder="e.g. Explain RAG in 3 bullet points", height=150)

    if st.button("⚡ Compare All 3 Models", type="primary", use_container_width=True):
        if not user_prompt.strip():
            st.warning("Please enter a prompt.")
        elif not all([openrouter_key, groq_key]):
            st.error("Please add both API keys in the sidebar.")
        else:
            col1, col2, col3 = st.columns(3)
            fns    = [call_gemini, call_llama, call_deepseek]
            keys   = [openrouter_key, groq_key, openrouter_key]
            colors = ["🔵", "🟠", "🟢"]
            results = []

            for col, fn, key, color in zip([col1, col2, col3], fns, keys, colors):
                with col:
                    with st.spinner(f"Calling {fn.__name__.replace('call_','')}..."):
                        r = fn(user_prompt, system_prompt, key)
                    results.append(r)
                    if r["error"]:
                        st.error(f"**{color} {r['model']}**\n\n❌ {r['error']}")
                    else:
                        st.success(f"**{color} {r['model']}**")
                        st.markdown(r["response"])
                        st.caption(f"⏱ {r['latency_ms']}ms | 📥 {r['input_tokens']} in | 📤 {r['output_tokens']} out")

            st.divider()
            st.subheader("Quick Comparison")
            cdf = pd.DataFrame([{
                "Model": r["model"], "Latency (ms)": r["latency_ms"],
                "Input Tokens": r["input_tokens"], "Output Tokens": r["output_tokens"],
                "Status": "✅" if not r["error"] else "❌"
            } for r in results])
            st.dataframe(cdf, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Run 10 standard benchmark prompts")
    st.caption("Makes 30 API calls · ~2–3 minutes")

    PROMPTS = [
        {"id":"P01","category":"Summarization","prompt":"Summarize in 3 bullet points: AI is transforming industries by automating tasks, enabling faster data analysis, and creating personalized experiences. It raises concerns about job displacement, privacy, and bias."},
        {"id":"P02","category":"Summarization","prompt":"Customer complaint: 'Ordered laptop 2 weeks ago. Late delivery, damaged box, dead pixels on screen. Want a refund.' Summarize and suggest 3 resolution steps."},
        {"id":"P03","category":"Q&A Factual","prompt":"What is the difference between supervised and unsupervised learning? Give one real-world example of each."},
        {"id":"P04","category":"Q&A Factual","prompt":"Explain the CAP theorem in distributed systems. How does it affect the design of a real-time chat application?"},
        {"id":"P05","category":"Classification","prompt":"Classify sentiment as Positive/Negative/Neutral with confidence High/Medium/Low: 'Product works fine but packaging was terrible and delivery took forever.'"},
        {"id":"P06","category":"Classification","prompt":"Classify into Billing/Technical/Account/Shipping and assign priority High/Medium/Low: 'Payment went through twice but I placed only one order. Fix urgently.'"},
        {"id":"P07","category":"Structured Output","prompt":"Return ONLY valid JSON {role, company, required_skills, experience_years, location}: 'Anthropic hiring Senior ML Engineer San Francisco, 5+ years, Python, PyTorch, distributed training.'"},
        {"id":"P08","category":"Structured Output","prompt":"Return ONLY valid JSON {product_name, target_user, top_3_features, biggest_risk, recommended_tech_stack} for: AI tool helping junior doctors identify drug interactions before prescribing."},
        {"id":"P09","category":"Reasoning","prompt":"Startup: $50K runway, $18K/month burn. Deal A: $30K 80% 1 month. Deal B: $60K 40% 2 months. Deal C: $15K 95% 3 weeks. Which to prioritize? Step-by-step reasoning."},
        {"id":"P10","category":"Reasoning","prompt":"PM with 2-week sprint. Features: A (high, 3d), B (medium, 5d), C (low, 1d), D (high, 8d). Cannot ship all. What to ship and why? Step by step."},
    ]

    if st.button("🚀 Run Full Benchmark", type="primary", use_container_width=True):
        if not all([openrouter_key, groq_key]):
            st.error("Please add both API keys in the sidebar.")
        else:
            all_results = []
            progress = st.progress(0, text="Starting...")
            total = len(PROMPTS) * 3
            fns  = [call_gemini, call_llama, call_deepseek]
            keys = [openrouter_key, groq_key, openrouter_key]
            done = 0

            for p in PROMPTS:
                for fn, key in zip(fns, keys):
                    done += 1
                    progress.progress(done/total, text=f"[{done}/{total}] {fn.__name__} on {p['id']}...")
                    r = fn(p["prompt"], "You are a helpful assistant.", key)
                    r.update({"prompt_id": p["id"], "category": p["category"]})
                    all_results.append(r)
                    time.sleep(0.8)

            progress.empty()
            bdf = pd.DataFrame(all_results)
            st.success(f"✅ Done! {len(bdf)} results. Errors: {bdf['error'].notna().sum()}")

            summary = bdf.groupby("model").agg(
                avg_latency=("latency_ms","mean"),
                avg_output =("output_tokens","mean"),
                errors     =("error", lambda x: x.notna().sum())
            ).round(1)
            st.subheader("Summary")
            st.dataframe(summary, use_container_width=True)

            st.subheader("All Responses")
            for pid in [p["id"] for p in PROMPTS]:
                rows = bdf[bdf["prompt_id"]==pid]
                with st.expander(f"{pid} — {rows.iloc[0]['category']}"):
                    c1, c2, c3 = st.columns(3)
                    for col, (_, row) in zip([c1,c2,c3], rows.iterrows()):
                        with col:
                            st.markdown(f"**{row['model']}**")
                            st.caption(f"{row['latency_ms']}ms | {row['output_tokens']} tokens")
                            if row["error"]:
                                st.error(row["error"])
                            else:
                                st.write(row["response"])

            csv = bdf.to_csv(index=False)
            st.download_button("⬇️ Download CSV", csv, "benchmark_results.csv",
                               "text/csv", use_container_width=True)