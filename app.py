import os
import io
import contextlib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import PyPDF2
import docx
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

# ==========================================
# 1. PAGE CONFIGURATION & ADVANCED CSS UI
# ==========================================
st.set_page_config(
    page_title="Nexus AI | Universal Data Agent",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

def apply_modern_ui():
    """Injects premium Glassmorphism and sleek modern UI styles."""
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Base typography */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Headers */
        h1, h2, h3 { color: #0F172A; font-weight: 800; letter-spacing: -0.5px; }
        
        /* Custom Metric Cards */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            transition: all 0.3s ease;
        }
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            border-color: #3B82F6;
        }
        
        /* Stylish Tabs */
        .stTabs [data-baseweb="tab-list"] { 
            gap: 12px; 
            padding-bottom: 5px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 600;
            color: #64748B;
            border: 1px solid #E2E8F0;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { background-color: #F1F5F9; color: #0F172A; }
        .stTabs [aria-selected="true"] { 
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            color: white !important; 
            border: none;
            box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        .stButton>button:hover { transform: scale(1.02); }
        
        /* Dataframe styling */
        .stDataFrame { border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        </style>
    """, unsafe_allow_html=True)

apply_modern_ui()

# ==========================================
# 2. UNIVERSAL FILE PARSING MODULE
# ==========================================
@st.cache_data(show_spinner=False)
def process_file(file):
    """Dynamically routes files based on extension and converts to a Pandas DataFrame."""
    try:
        ext = file.name.split('.')[-1].lower()
        
        # 1. CSV
        if ext == 'csv':
            return pd.read_csv(file)
            
        # 2. Excel
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(file)
            
        # 3. PDF Parsing
        elif ext == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            data = []
            for i, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    data.append({
                        "Page_Number": i + 1,
                        "Text_Content": text.strip(),
                        "Word_Count": len(text.split()),
                        "Char_Count": len(text)
                    })
            if not data:
                return "Error: Could not extract readable text from PDF. It might be scanned/image-based."
            return pd.DataFrame(data)
            
        # 4. Word Document Parsing
        elif ext == 'docx':
            doc = docx.Document(file)
            data = []
            for i, para in enumerate(doc.paragraphs):
                text = para.text.strip()
                if text:
                    data.append({
                        "Paragraph_ID": i + 1,
                        "Text_Content": text,
                        "Word_Count": len(text.split()),
                        "Char_Count": len(text)
                    })
            if not data:
                return "Error: The Word document is empty."
            return pd.DataFrame(data)
            
        else:
            return f"Error: Unsupported file format '.{ext}'"
            
    except Exception as e:
        return f"File processing failed: {str(e)}"

# ==========================================
# 3. LLM INITIALIZATION
# ==========================================
def get_llm(model_name, provider, api_key):
    """Initializes the LLM based on user selection securely."""
    if provider == "google_genai":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif provider == "groq":
        os.environ["GROQ_API_KEY"] = api_key
        
    return init_chat_model(model_name, model_provider=provider)

# ==========================================
# 4. SIDEBAR & CONFIGURATION
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🌌 Nexus AI</h2>", unsafe_allow_html=True)
    st.caption("<div style='text-align: center;'>Universal Data & Document Agent</div>", unsafe_allow_html=True)
    st.divider()
    
    with st.expander("🔑 1. API Configuration", expanded=True):
        provider_choice = st.selectbox("Select AI Engine", ["Google Gemini", "Groq"])
        
        if provider_choice == "Google Gemini":
            api_key = st.text_input("Google API Key", type="password", placeholder="AIzaSy...")
            model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-1.5-pro"])
            provider = "google_genai"
        else:
            api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
            model_name = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"])
            provider = "groq"
            
    with st.expander("📂 2. File Upload", expanded=True):
        st.markdown("**Supported:** CSV, Excel, PDF, Word (Docx)")
        uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "xls", "pdf", "docx"], label_visibility="collapsed")
        
    st.divider()
    st.markdown("### 💡 Pro Tips:")
    st.info("- **PDFs/Word Docs** are automatically converted into datasets (rows of text/pages) so you can plot their word counts or chat with their content!")

# ==========================================
# 5. MAIN APPLICATION LOGIC
# ==========================================
if uploaded_file is None:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem;'>Welcome to Nexus AI 🚀</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 1.2rem;'>Upload a dataset or document in the sidebar to magically analyze it.</p>", unsafe_allow_html=True)
    st.stop()

# State Management for Data Modifications
with st.spinner(f"Extracting and Structuring {uploaded_file.name}..."):
    raw_data = process_file(uploaded_file)

if isinstance(raw_data, str):
    st.error(raw_data)
    st.stop()

if "df" not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
    st.session_state.df = raw_data.copy()
    st.session_state.last_file = uploaded_file.name

working_df = st.session_state.df

# Header Section
colA, colB = st.columns([3, 1])
with colA:
    st.title(f"📊 {uploaded_file.name}")
    file_ext = uploaded_file.name.split('.')[-1].upper()
    st.caption(f"**Format:** {file_ext} | **Engine:** {model_name} via {provider_choice}")
with colB:
    # Download processed data feature
    csv_data = working_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Download Processed Data",
        data=csv_data,
        file_name=f"processed_{uploaded_file.name}.csv",
        mime='text/csv',
        use_container_width=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# Application Tabs
tab_clean, tab_eda, tab_viz, tab_ai_code, tab_chat = st.tabs([
    "🧹 Data Prep", 
    "📋 Smart EDA", 
    "📈 Interactive Viz", 
    "⚡ AI Code", 
    "💬 Chat w/ Data"
])

# ------------------------------------------
# TAB 1: DATA PREPARATION & CLEANING
# ------------------------------------------
with tab_clean:
    st.header("Data Preparation Engine")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Missing Values**")
        if st.button("Drop Missing Values (NaN)", use_container_width=True):
            st.session_state.df = working_df.dropna()
            st.success("Successfully dropped rows with missing values!")
            st.rerun()
            
    with c2:
        st.markdown("**Duplicates**")
        if st.button("Drop Duplicate Rows", use_container_width=True):
            st.session_state.df = working_df.drop_duplicates()
            st.success("Successfully dropped duplicate rows!")
            st.rerun()
            
    with c3:
        st.markdown("**Restore**")
        if st.button("Reset to Original File", use_container_width=True, type="secondary"):
            st.session_state.df = process_file(uploaded_file)
            st.success("Data reset to original state!")
            st.rerun()
            
    st.divider()
    st.subheader("Data Preview")
    st.dataframe(working_df.head(50), use_container_width=True)

# ------------------------------------------
# TAB 2: AUTOMATED EDA
# ------------------------------------------
with tab_eda:
    st.header("Exploratory Data Analysis")
    
    # Hero Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows / Entries", f"{working_df.shape[0]:,}")
    m2.metric("Columns / Features", f"{working_df.shape[1]:,}")
    m3.metric("Missing Data Points", f"{working_df.isnull().sum().sum():,}")
    m4.metric("Duplicate Rows", f"{working_df.duplicated().sum():,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("🗂️ Schema Details")
        dtype_df = pd.DataFrame(working_df.dtypes, columns=['Data Type']).reset_index()
        dtype_df.rename(columns={'index': 'Feature Name'}, inplace=True)
        dtype_df['Data Type'] = dtype_df['Data Type'].astype(str)
        st.dataframe(dtype_df, use_container_width=True, hide_index=True, height=250)
        
    with col2:
        st.subheader("⚠️ Missing Value Analysis")
        missing_df = working_df.isnull().sum().reset_index()
        missing_df.columns = ['Feature Name', 'Missing Count']
        missing_df['Missing (%)'] = (missing_df['Missing Count'] / len(working_df)) * 100
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)
        
        if missing_df.empty:
            st.success("🎉 Incredible! Your dataset is perfectly clean with no missing values.")
        else:
            st.dataframe(missing_df.style.format({'Missing (%)': '{:.2f}%'}), use_container_width=True, hide_index=True, height=250)
            
    st.divider()
    
    # Statistical Breakdowns
    num_cols = working_df.select_dtypes(include=['number'])
    cat_cols = working_df.select_dtypes(include=['object', 'category'])
    
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.subheader("📈 Numeric Distributions")
        if not num_cols.empty:
            st.dataframe(num_cols.describe().T, use_container_width=True)
        else:
            st.info("No numerical features detected.")
            
    with stat_col2:
        st.subheader("🔡 Categorical Summaries")
        if not cat_cols.empty:
            st.dataframe(cat_cols.describe().T, use_container_width=True)
        else:
            st.info("No text/categorical features detected.")

# ------------------------------------------
# TAB 3: INTERACTIVE VISUALIZATIONS
# ------------------------------------------
with tab_viz:
    st.header("Visual Analytics Studio")
    
    all_cols = working_df.columns.tolist()
    num_col_list = working_df.select_dtypes(include=['number']).columns.tolist()
    
    v1, v2 = st.columns([1, 3])
    
    with v1:
        st.markdown("#### Studio Controls")
        chart_type = st.selectbox("Chart Type", [
            "Scatter Plot", "Bar Chart", "Line Trend", 
            "Histogram", "Box Plot", "Violin Plot", "Correlation Matrix"
        ])
        
        if chart_type != "Correlation Matrix":
            x_axis = st.selectbox("X-Axis Feature", all_cols)
            
            # Default Y to the first numeric column if it exists and isn't the X axis
            default_y_index = 0
            if len(all_cols) > 1:
                default_y_index = 1 if all_cols[0] == x_axis else 0
                
            y_axis = st.selectbox("Y-Axis Feature", all_cols, index=default_y_index)
            color_by = st.selectbox("Group / Color By", ["None"] + all_cols)
        
    with v2:
        st.markdown("#### Render Output")
        try:
            if chart_type == "Correlation Matrix":
                if len(num_col_list) >= 2:
                    corr = working_df[num_col_list].corr()
                    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="Viridis", title="Numeric Feature Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Heatmap requires at least 2 numeric features. (e.g., Word Count & Character Count for PDFs)")
            else:
                c_arg = color_by if color_by != "None" else None
                
                # Plotly dynamic routing
                if chart_type == "Scatter Plot":
                    fig = px.scatter(working_df, x=x_axis, y=y_axis, color=c_arg, title=f"Scatter: {y_axis} vs {x_axis}", template="plotly_white")
                elif chart_type == "Bar Chart":
                    # Smart aggregation to prevent browser crash on huge data
                    agg_df = working_df.groupby(x_axis, as_index=False)[y_axis].sum() if y_axis in num_col_list else working_df
                    fig = px.bar(agg_df, x=x_axis, y=y_axis, color=c_arg, title=f"Bar Aggregation: {y_axis} by {x_axis}", template="plotly_white")
                elif chart_type == "Line Trend":
                    fig = px.line(working_df, x=x_axis, y=y_axis, color=c_arg, title=f"Trend: {y_axis} over {x_axis}", template="plotly_white")
                elif chart_type == "Histogram":
                    fig = px.histogram(working_df, x=x_axis, color=c_arg, title=f"Distribution of {x_axis}", template="plotly_white")
                elif chart_type == "Box Plot":
                    fig = px.box(working_df, x=x_axis, y=y_axis, color=c_arg, title=f"Box Spread: {y_axis} by {x_axis}", template="plotly_white")
                elif chart_type == "Violin Plot":
                    fig = px.violin(working_df, x=x_axis, y=y_axis, color=c_arg, box=True, title=f"Violin Spread: {y_axis} by {x_axis}", template="plotly_white")
                
                # Apply aesthetic tweaks
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Render Error: {str(e)}")

# ------------------------------------------
# TAB 4: AI CODE EXECUTOR
# ------------------------------------------
with tab_ai_code:
    st.header("⚡ AI Code Studio")
    st.info("Write a prompt. The AI generates Python code (using Pandas/Matplotlib/Seaborn/Plotly) and executes it against your data in real-time.")
    
    user_prompt = st.text_area("What analytical operation or plot do you want to run?", 
                               "Create a sophisticated seaborn pairplot for the numerical features.",
                               height=100)
    
    if st.button("Generate & Stage Code", type="primary"):
        if not api_key:
            st.error("⚠️ API Key required. Please configure it in the sidebar.")
        else:
            with st.spinner("🤖 Writing custom Python script..."):
                try:
                    chat = get_llm(model_name, provider, api_key)
                    
                    sys_prompt = (
                        "You are an Elite Python Data Scientist. The user has loaded a Pandas DataFrame named `df`. "
                        "Write Python code to fulfill their request. Use pandas, matplotlib.pyplot as plt, seaborn as sns, or plotly.express as px. "
                        "If you create a matplotlib/seaborn plot, end with `plt.show()`. If you use plotly, end with `fig.show()`. "
                        "OUTPUT STRICTLY VALID PYTHON CODE. DO NOT INCLUDE MARKDOWN TICK MARKS (```) OR EXPLANATIONS."
                    )
                    
                    data_context = f"Schema:\nColumns: {list(working_df.columns)}\nData Types:\n{working_df.dtypes.to_string()}"
                    
                    messages = [
                        SystemMessage(content=sys_prompt),
                        HumanMessage(content=f"{data_context}\n\nTask: {user_prompt}")
                    ]
                    
                    response = chat.invoke(messages)
                    
                    # Clean the AI output just in case it disobeys the instruction
                    clean_code = response.content.replace("```python", "").replace("```", "").strip()
                    st.session_state.staged_code = clean_code
                    
                except Exception as e:
                    st.error(f"Code Generation Failed: {e}")

    # Execution Area
    if "staged_code" in st.session_state:
        st.markdown("### 🔍 Review Script")
        # Allow user to edit code before executing (Crucial for safety)
        editable_code = st.text_area("Python Script", value=st.session_state.staged_code, height=250)
        
        if st.button("▶️ Execute Script", use_container_width=True):
            st.markdown("### 🖥️ Console Output")
            buffer = io.StringIO()
            with st.spinner("Executing runtime..."):
                try:
                    # Provide an isolated environment
                    local_vars = {"df": working_df, "pd": pd, "plt": plt, "sns": sns, "px": px, "go": go}
                    
                    # Capture printed output
                    with contextlib.redirect_stdout(buffer):
                        exec(editable_code, local_vars)
                        
                    output = buffer.getvalue()
                    if output:
                        st.code(output, language="bash")
                        
                    # Capture matplotlib plots if generated
                    if plt.get_fignums():
                        st.pyplot(plt.gcf())
                        plt.clf() # Clear figure for next run
                        
                except Exception as run_err:
                    st.error(f"Runtime Exception: {run_err}")

# ------------------------------------------
# TAB 5: CHAT WITH DATA
# ------------------------------------------
with tab_chat:
    st.header("💬 Conversational Data Agent")
    st.caption("Ask questions about your data. The AI will analyze the structure, summary statistics, and content to answer you.")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Render historical chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Input Area
    if chat_prompt := st.chat_input("Ask about patterns, summaries, or specific records..."):
        if not api_key:
            st.error("⚠️ API Key required. Please configure it in the sidebar.")
        else:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": chat_prompt})
            with st.chat_message("user"):
                st.markdown(chat_prompt)
                
            # Process AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing dataset logic..."):
                    try:
                        chat = get_llm(model_name, provider, api_key)
                        
                        # Build optimal context window (truncate huge datasets)
                        head_sample = working_df.head(5).to_csv(index=False)
                        desc_stats = working_df.describe(include='all').to_csv()
                        
                        context = (
                            f"System Context: You are a helpful AI Data Analyst.\n"
                            f"Dataset Shape: {working_df.shape[0]} rows, {working_df.shape[1]} cols.\n"
                            f"First 5 Rows:\n{head_sample}\n\n"
                            f"Summary Stats:\n{desc_stats}\n"
                        )
                        
                        history = [SystemMessage(content=context)]
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
                        st.error(f"Inference Error: {e}")
