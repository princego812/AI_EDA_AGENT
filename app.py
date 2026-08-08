import os
import io
import contextlib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="DataSense AI | Smart Analyst",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_custom_css():
    st.markdown("""
        <style>
        /* Main Theme Adjustments */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        /* Custom Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            padding: 5% 5% 5% 10%;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            transition: transform 0.2s ease-in-out;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 4px 4px 15px rgba(0,0,0,0.1);
        }
        
        /* Style Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #F1F5F9;
            border-radius: 6px 6px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] { background-color: #1E3A8A; color: white; }
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
@st.cache_data(show_spinner=False)
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(file)
    except Exception as e:
        return str(e)
    return None

def get_llm(model_name, provider, api_key):
    """Initializes the LLM based on user selection."""
    if provider == "google_genai":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "groq":
        os.environ["GROQ_API_KEY"] = api_key
        
    return init_chat_model(model_name, model_provider=provider)

# ==========================================
# 3. SIDEBAR UI & SETUP
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103832.png", width=60)
    st.title("DataSense AI")
    st.caption("Your Automated AI Data Analyst")
    st.divider()
    
    st.header("1. API Configuration")
    provider_choice = st.selectbox("Select AI Provider", ["Google Gemini", "Groq"])
    
    if provider_choice == "Google Gemini":
        api_key = st.text_input("Google API Key", type="password", help="Get it from Google AI Studio")
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
        provider = "google_genai"
    else:
        api_key = st.text_input("Groq API Key", type="password", help="Get it from Groq Cloud")
        model_name = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"])
        provider = "groq"
        
    st.divider()
    st.header("2. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

# ==========================================
# 4. MAIN DASHBOARD LOGIC
# ==========================================
if uploaded_file is None:
    st.markdown("<h1 style='text-align: center; color: #64748b;'>Welcome to DataSense AI 🚀</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8;'>Please upload a CSV or Excel dataset in the sidebar to begin analysis.</p>", unsafe_allow_html=True)
    st.stop()

# Load Data
df = load_data(uploaded_file)
if isinstance(df, str):
    st.error(f"Error loading file: {df}")
    st.stop()

# Store dataframe in session state so cleaning operations persist
if "df" not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
    st.session_state.df = df
    st.session_state.last_file = uploaded_file.name

working_df = st.session_state.df

st.title(f"📊 Dataset: {uploaded_file.name}")
st.caption(f"Powered by {model_name} via {provider_choice}")

# TABS DEFINITION
tab_clean, tab_eda, tab_viz, tab_ai_code, tab_chat = st.tabs([
    "🧹 Data Cleaning", 
    "📋 Smart EDA", 
    "📈 Interactive Visualizations", 
    "⚡ AI Code Executor", 
    "💬 Chat w/ Data"
])

# ------------------------------------------
# TAB 1: DATA CLEANING
# ------------------------------------------
with tab_clean:
    st.subheader("Data Cleaning & Preprocessing")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Handle Missing Values**")
        if st.button("Drop Rows with Missing Values"):
            st.session_state.df = working_df.dropna()
            st.success("Dropped missing values!")
            st.rerun()
            
    with c2:
        st.markdown("**Handle Duplicates**")
        if st.button("Drop Duplicate Rows"):
            st.session_state.df = working_df.drop_duplicates()
            st.success("Dropped duplicates!")
            st.rerun()
            
    with c3:
        st.markdown("**Reset Data**")
        if st.button("Revert to Original File"):
            st.session_state.df = load_data(uploaded_file)
            st.success("Reverted to original data!")
            st.rerun()
            
    st.divider()
    st.markdown("**Current Data Preview**")
    st.dataframe(working_df.head(10), use_container_width=True)

# ------------------------------------------
# TAB 2: AUTOMATED EDA
# ------------------------------------------
with tab_eda:
    st.subheader("Exploratory Data Analysis (EDA)")
    
    # High-level Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{working_df.shape[0]:,}")
    m2.metric("Total Columns", f"{working_df.shape[1]:,}")
    m3.metric("Missing Values", f"{working_df.isnull().sum().sum():,}")
    m4.metric("Duplicate Rows", f"{working_df.duplicated().sum():,}")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 🗂️ Column Data Types")
        dtype_df = pd.DataFrame(working_df.dtypes, columns=['Data Type']).reset_index()
        dtype_df.rename(columns={'index': 'Column Name'}, inplace=True)
        dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        
    with col2:
        st.markdown("### ⚠️ Missing Values Breakdown")
        missing_df = working_df.isnull().sum().reset_index()
        missing_df.columns = ['Column Name', 'Missing Count']
        missing_df['Missing %'] = (missing_df['Missing Count'] / len(working_df)) * 100
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
        
        if missing_df.empty:
            st.success("🎉 No missing values found in the dataset!")
        else:
            st.dataframe(missing_df.style.format({'Missing %': '{:.2f}%'}), use_container_width=True, hide_index=True)
            
    st.markdown("### 📈 Statistical Summary (Numerical)")
    num_df = working_df.select_dtypes(include=['number'])
    if not num_df.empty:
        st.dataframe(num_df.describe().T, use_container_width=True)
    else:
        st.info("No numerical columns available.")

    st.markdown("### 🔡 Statistical Summary (Categorical)")
    cat_df = working_df.select_dtypes(include=['object', 'category'])
    if not cat_df.empty:
        st.dataframe(cat_df.describe().T, use_container_width=True)
    else:
        st.info("No categorical columns available.")

# ------------------------------------------
# TAB 3: INTERACTIVE VISUALIZATIONS
# ------------------------------------------
with tab_viz:
    st.subheader("No-Code Interactive Visualizations")
    num_cols = working_df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = working_df.select_dtypes(include=['object', 'category']).columns.tolist()
    all_cols = working_df.columns.tolist()
    
    v1, v2 = st.columns([1, 3])
    
    with v1:
        st.markdown("#### Chart Settings")
        chart_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Chart", "Line Chart", "Histogram", "Box Plot", "Correlation Heatmap"])
        
        if chart_type != "Correlation Heatmap":
            x_axis = st.selectbox("X-Axis", all_cols)
            y_axis = st.selectbox("Y-Axis", all_cols, index=1 if len(all_cols) > 1 else 0)
            color_by = st.selectbox("Color By (Optional)", ["None"] + all_cols)
        
    with v2:
        st.markdown("#### Visualization Output")
        try:
            if chart_type == "Correlation Heatmap":
                if len(num_cols) > 1:
                    corr = working_df[num_cols].corr()
                    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Need at least 2 numerical columns for a heatmap.")
            else:
                color_arg = color_by if color_by != "None" else None
                
                if chart_type == "Scatter Plot":
                    fig = px.scatter(working_df, x=x_axis, y=y_axis, color=color_arg, title=f"{y_axis} vs {x_axis}")
                elif chart_type == "Bar Chart":
                    # Grouping data for bar charts to prevent crashing on large datasets
                    bar_data = working_df.groupby(x_axis, as_index=False)[y_axis].sum() if y_axis in num_cols else working_df
                    fig = px.bar(bar_data, x=x_axis, y=y_axis, color=color_arg, title=f"Bar Chart: {y_axis} by {x_axis}")
                elif chart_type == "Line Chart":
                    fig = px.line(working_df, x=x_axis, y=y_axis, color=color_arg, title=f"Trend of {y_axis} over {x_axis}")
                elif chart_type == "Histogram":
                    fig = px.histogram(working_df, x=x_axis, color=color_arg, title=f"Distribution of {x_axis}")
                elif chart_type == "Box Plot":
                    fig = px.box(working_df, x=x_axis, y=y_axis, color=color_arg, title=f"Box Plot: {y_axis} across {x_axis}")
                
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate plot. Error: {str(e)}")

# ------------------------------------------
# TAB 4: AI CODE EXECUTOR
# ------------------------------------------
with tab_ai_code:
    st.subheader("⚡ AI Code Generator & Executor")
    st.info("Ask the AI to analyze, manipulate, or plot the data. It will write Python code and execute it directly!")
    
    user_prompt = st.text_area("What would you like to do with the data?", "Plot a seaborn distribution plot of the first numerical column.")
    
    if st.button("Generate Code", type="primary"):
        if not api_key:
            st.error("Please provide your API key in the sidebar.")
        else:
            with st.spinner("🧠 AI is thinking and writing code..."):
                try:
                    chat = get_llm(model_name, provider, api_key)
                    
                    sys_msg = (
                        "You are a Senior Python Data Scientist. A pandas DataFrame `df` is already loaded in memory. "
                        "Write executable Python code to fulfill the user's request using `df`. "
                        "Use matplotlib, seaborn, or pandas for plotting (use plt.show() if plotting). "
                        "DO NOT include markdown, explanations, or backticks (```python) in your output. RETURN RAW CODE ONLY."
                    )
                    
                    data_context = f"Columns: {list(working_df.columns)}\nTypes: {working_df.dtypes.to_dict()}"
                    
                    messages = [
                        SystemMessage(content=sys_msg),
                        HumanMessage(content=f"Context:\n{data_context}\n\nRequest: {user_prompt}")
                    ]
                    
                    response = chat.invoke(messages)
                    generated_code = response.content.replace("```python", "").replace("```", "").strip()
                    
                    st.session_state.generated_code = generated_code
                except Exception as e:
                    st.error(f"AI Generation Error: {e}")

    if "generated_code" in st.session_state:
        st.markdown("### Review & Edit Code")
        # Allow the user to edit the AI's code before execution for safety
        editable_code = st.text_area("Python Code", value=st.session_state.generated_code, height=200)
        
        if st.button("▶️ Execute Code"):
            st.markdown("### Execution Output:")
            buffer = io.StringIO()
            with st.spinner("Executing..."):
                try:
                    # Provide an isolated environment containing only necessary globals
                    local_vars = {"df": working_df, "pd": pd, "plt": plt, "sns": sns, "px": px}
                    
                    # Capture stdout
                    with contextlib.redirect_stdout(buffer):
                        exec(editable_code, local_vars)
                        
                    output_result = buffer.getvalue()
                    if output_result:
                        st.code(output_result)
                        
                    # Capture Matplotlib figures
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.clf()
                        
                except Exception as exec_error:
                    st.error(f"Execution Error: {exec_error}")

# ------------------------------------------
# TAB 5: CHAT WITH DATA
# ------------------------------------------
with tab_chat:
    st.subheader("💬 Chat with your Dataset")
    st.caption("Ask questions about trends, meanings, or summaries based on your data.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    if chat_prompt := st.chat_input("Ask a question (e.g., 'What are the key takeaways from this data?'):"):
        if not api_key:
            st.error("Please provide your API key in the sidebar to use Chat.")
        else:
            # Add user message to UI
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            with st.chat_message("user"):
                st.markdown(chat_prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        chat = get_llm(model_name, provider, api_key)
                        
                        # Build compact context to avoid token limits
                        head_data = working_df.head(5).to_csv(index=False)
                        desc_data = working_df.describe().to_csv()
                        context_str = f"Dataset Shape: {working_df.shape}\nFirst 5 rows:\n{head_data}\nSummary Stats:\n{desc_data}"
                        
                        system_msg = SystemMessage(
                            content=f"You are a helpful Data Analyst. Use the following dataset context to answer the user's queries accurately.\nContext:\n{context_str}"
                        )
                        
                        history = [system_msg]
                        for m in st.session_state.messages:
                            if m["role"] == "user":
                                history.append(HumanMessage(content=m["content"]))
                            else:
                                history.append(SystemMessage(content=m["content"]))
                                
                        response = chat.invoke(history)
                        ai_reply = response.content
                        st.markdown(ai_reply)
                        
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    except Exception as e:
                        st.error(f"Chat Error: {e}")
