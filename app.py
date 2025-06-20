import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import docx
import PyPDF2
import re
from io import StringIO, BytesIO
import google.generativeai as genai
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import io

# --- Gemini Configuration ---
GOOGLE_API_KEY = "YOUR_VALID_API_KEY_HERE"  # Replace with a valid Google API key
genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to extract text from PDF (machine-printed)
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Function to extract text from image using Tesseract OCR
def extract_text_from_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang='eng')
        return text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

# Function to extract text from PDF using OCR (for handwritten PDFs)
def extract_text_from_pdf_ocr(file):
    try:
        images = convert_from_bytes(file.read())
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image, lang='eng') + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF with OCR: {str(e)}"

# Function to parse supplier data
def parse_supplier_data(text):
    lines = text.split("\n")
    supplier_name = "Unknown Supplier"
    total_budget = 0
    materials = {}
    
    # Regex patterns
    cost_pattern = r'(?:•|\-|\s|^)\s([a-zA-Z\s]+)\s*:\s*RS\s*([\d,]+)(?:\s|$)'  # Matches "• Wooden Works: Rs 20,000"
    total_pattern = r"total\s*(?:budget|predicted budget|amounts to)\s*(?:is)?\s*RS\.?\s*([\d,]+)(?:\.|\s|$)"  # Matches "The total predicted budget is Rs 72,000."
    
    # Extract supplier name
    for line in lines:
        line = line.strip()
        if not line:
            continue
        supplier_match = re.search(r"^(?:Supplier\s*[:|-]?\s*)?([A-Za-z\s]+?)(?:We are the|[:,]|\s*$)", line, re.IGNORECASE)
        if supplier_match:
            supplier_name = supplier_match.group(1).strip()
            break
        if supplier_name == "Unknown Supplier":
            supplier_name = line.strip()
            break
    
    # Extract material costs and total budget
    for line in lines:
        line = line.strip()
        # Extract material costs
        costs = re.findall(cost_pattern, line, re.IGNORECASE)
        for material, cost_str in costs:
            materials[material.strip()] = int(cost_str.replace(",", ""))
        
        # Extract total budget
        total_match = re.search(total_pattern, line, re.IGNORECASE)
        if total_match:
            total_budget = int(total_match.group(1).replace(",", ""))
    
    return {
        "Supplier": supplier_name,
        "Materials": materials,
        "Total Budget": total_budget
    }

# Function to process uploaded files and store in session state
def process_files(uploaded_files, round_key):
    data = []
    if f"{round_key}_data" not in st.session_state:
        st.session_state[f"{round_key}_data"] = {}
    
    for file in uploaded_files:
        file_name = file.name.lower()
        try:
            if file_name.endswith(".docx"):
                text = extract_text_from_docx(file)
            elif file_name.endswith(".pdf"):
                text = extract_text_from_pdf(file)
                if not text.strip():
                    file.seek(0)
                    text = extract_text_from_pdf_ocr(file)
            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                text = extract_text_from_image(file)
            elif file_name.endswith(".csv"):
                text = file.read().decode("utf-8")
            else:
                st.warning(f"Unsupported file format: {file_name}")
                continue
            supplier_data = parse_supplier_data(text)
            supplier_data["Round"] = round_key
            data.append(supplier_data)
            st.session_state[f"{round_key}_data"][file_name] = supplier_data
        except Exception as e:
            st.error(f"Error processing {file_name}: {str(e)}")
    return data

# Function to answer questions about invoices using Gemini
def ask_question_about_invoice(question, round_key=None):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        context = "You are a smart assistant. Use the following invoice data to answer the user's question:\n\n"
        if round_key:
            invoices_data = st.session_state.get(f"{round_key}_data", {})
            context += f"Round: {round_key}\n"
            for file_name, data in invoices_data.items():
                context += f"Invoice: {file_name}\nSupplier: {data['Supplier']}\nTotal Budget: RS {data['Total Budget']}\nMaterials: {', '.join([f'{k} (RS {v})' for k, v in data['Materials'].items()]) if data['Materials'] else 'None'}\n\n"
        else:
            for round_key in ["Round 1", "Round 2"]:
                invoices_data = st.session_state.get(f"{round_key}_data", {})
                context += f"Round: {round_key}\n"
                for file_name, data in invoices_data.items():
                    context += f"Invoice: {file_name}\nSupplier: {data['Supplier']}\nTotal Budget: RS {data['Total Budget']}\nMaterials: {', '.join([f'{k} (RS {v})' for k, v in data['Materials'].items()]) if data['Materials'] else 'None'}\n\n"
        context += f"Question: {question}"
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        return f"Error processing question: {str(e)}. Please try again or rephrase your question."

# Function to create a DataFrame from supplier data
def create_dataframe(data):
    suppliers = []
    material_types = set()
    for supplier in data:
        suppliers.append(supplier["Supplier"])
        material_types.update(supplier["Materials"].keys())
    
    df = pd.DataFrame(columns=["Supplier", "Round"] + sorted(list(material_types)) + ["Total Budget"])
    for i, supplier in enumerate(data):
        row = {"Supplier": supplier["Supplier"], "Round": supplier["Round"], "Total Budget": supplier["Total Budget"]}
        for material in material_types:
            row[material] = supplier["Materials"].get(material, 0)
        df.loc[i] = row
    # Ensure Total Budget is numeric and handle NaN
    df["Total Budget"] = pd.to_numeric(df["Total Budget"], errors='coerce').fillna(0).astype(int)
    return df

# Function to generate Excel file
def generate_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Supplier Data', index=False)
    return output.getvalue()

# Function to generate Excel file for transposed table
def generate_transposed_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Transposed Supplier Data', index=True)
    return output.getvalue()

# Function to compare Round 1 and Round 2
def compare_rounds(df):
    if df[df["Round"] == "Round 1"].empty or df[df["Round"] == "Round 2"].empty:
        return "Insufficient data to compare Round 1 and Round 2."
    avg_budget_r1 = df[df["Round"] == "Round 1"]["Total Budget"].mean()
    avg_budget_r2 = df[df["Round"] == "Round 2"]["Total Budget"].mean()
    diff = avg_budget_r2 - avg_budget_r1
    if diff < 0:
        return f"Round 2 average total budget is Rs {-diff:,.0f} lower than Round 1."
    elif diff > 0:
        return f"Round 2 average total budget is Rs {diff:,.0f} higher than Round 1."
    else:
        return "Round 1 and Round 2 have the same average total budget."

# AI-based recommendation logic
def recommend_best_supplier(df, round_key=None):
    if round_key:
        df = df[df["Round"] == round_key]
    if df.empty:
        return None, 0, None, df
    df["Material Coverage"] = df.drop(columns=["Supplier", "Round", "Total Budget"]).apply(lambda x: (x > 0).sum(), axis=1)
    max_coverage = df["Material Coverage"].max() if df["Material Coverage"].max() != 0 else 1
    min_budget = df["Total Budget"].min() if df["Total Budget"].min() != 0 else 1
    df["Score"] = (df["Material Coverage"] / max_coverage) / (df["Total Budget"] / min_budget)
    best_supplier = df.loc[df["Score"].idxmax()]
    return best_supplier["Supplier"], best_supplier["Score"], best_supplier["Round"], df

# Streamlit app
st.set_page_config(page_title="Construction Material Cost Comparison", layout="wide")
st.title("Construction Material Cost Comparison Dashboard")
st.markdown("Upload supplier documents for Round 1 and Round 2 bidding (DOCX, PDF, CSV, PNG, JPG, JPEG) to compare costs across materials, filter by suppliers, materials, supplier type, get AI-based recommendations, or ask questions about the invoices.")

# File uploaders for Round 1 and Round 2
st.header("Upload Bidding Documents")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Round 1 Bidding")
    uploaded_files_round1 = st.file_uploader("Upload Round 1 Documents", type=["docx", "pdf", "csv", "png", "jpg", "jpeg"], accept_multiple_files=True, key="round1_uploader")
with col2:
    st.subheader("Round 2 Bidding")
    uploaded_files_round2 = st.file_uploader("Upload Round 2 Documents", type=["docx", "pdf", "csv", "png", "jpg", "jpeg"], accept_multiple_files=True, key="round2_uploader")

# Process files
data_round1 = []
data_round2 = []
if uploaded_files_round1:
    data_round1 = process_files(uploaded_files_round1, "Round 1")
if uploaded_files_round2:
    data_round2 = process_files(uploaded_files_round2, "Round 2")

# Combine data
data = data_round1 + data_round2
if not data:
    st.info("Please upload at least one supplier document for Round 1 or Round 2 to begin the comparison.")
    st.stop()

# Create DataFrame
df = create_dataframe(data)

# Download Excel file for original table
st.subheader("Download Supplier Data")
excel_data = generate_excel(df)
st.download_button(
    label="Download Original Table as Excel",
    data=excel_data,
    file_name="supplier_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Filters
st.header("Filters")
col1, col2, col3 = st.columns(3)
with col1:
    round_filter = st.selectbox("Select Bidding Round", ["All", "Round 1", "Round 2"])
filtered_df = df.copy() if round_filter == "All" else df[df["Round"] == round_filter]

with col3:
    supplier_type = st.selectbox("Select Supplier Type", ["All", "Basic Needs (Foundation/Basement)", "Secondary Needs", "Interior/Decoration", "Household Necessities"])
if supplier_type != "All":
    type_keywords = {
        "Basic Needs (Foundation/Basement)": ["sand", "stone", "cement", "rods"],
        "Secondary Needs": ["paint", "woods", "door works", "pipes"],
        "Interior/Decoration": ["modular kitchen", "wall decorations", "taps and sink", "marble works"],
        "Household Necessities": ["cupboards", "drawers", "tables", "wooden works", "tiles", "door works", "steel work"]
    }
    keywords = type_keywords.get(supplier_type, [])
    filtered_df = filtered_df[filtered_df.apply(lambda row: any(row[key] > 0 for key in keywords if key in row), axis=1)]

with col2:
    material_cols = [col for col in filtered_df.columns if col not in ["Supplier", "Round", "Total Budget"]]
    valid_materials = [col for col in material_cols if filtered_df[col].sum() > 0]
    material_options = ["All"] + valid_materials
    selected_materials = st.multiselect("Select Materials", material_options, default=["All"])
material_filtered_df = filtered_df.copy()
if "All" not in selected_materials:
    material_filtered_df = filtered_df[filtered_df[selected_materials].sum(axis=1) > 0]

with col1:
    valid_suppliers = material_filtered_df["Supplier"].tolist()
    supplier_options = ["All"] + valid_suppliers
    previous_selected_suppliers = st.session_state.get("selected_suppliers", ["All"])
    valid_selected_suppliers = [s for s in previous_selected_suppliers if s in supplier_options]
    if not valid_selected_suppliers:
        valid_selected_suppliers = ["All"]
    selected_suppliers = st.multiselect("Select Suppliers", supplier_options, default=valid_selected_suppliers, key="supplier_filter")
    st.session_state["selected_suppliers"] = selected_suppliers

if "All" not in selected_suppliers:
    filtered_df = material_filtered_df[material_filtered_df["Supplier"].isin(selected_suppliers)]
else:
    filtered_df = material_filtered_df

# Display original Supplier Cost Comparison table
st.subheader("Supplier Cost Comparison")
table_columns = ["Supplier", "Round"] + (valid_materials if "All" in selected_materials else [col for col in selected_materials if col in filtered_df.columns]) + ["Total Budget"]
table_df = filtered_df[table_columns]
if not table_df.empty:
    st.dataframe(table_df)
else:
    st.write("No data matches the selected filters. Please adjust the filters to see the comparison table.")

# Create transposed table with Round 1 and Round 2 side by side for the same supplier
# Use the full DataFrame (df) to include both rounds, apply filters as needed
transposed_df = df.copy()
if supplier_type != "All":
    transposed_df = transposed_df[transposed_df.apply(lambda row: any(row[key] > 0 for key in keywords if key in row), axis=1)]
if "All" not in selected_materials:
    transposed_df = transposed_df[transposed_df[selected_materials].sum(axis=1) > 0]
if "All" not in selected_suppliers:
    transposed_df = transposed_df[transposed_df["Supplier"].isin(selected_suppliers)]

# Create unique column names by combining Supplier and Round
transposed_df["Supplier_Round"] = transposed_df["Supplier"] + " (" + transposed_df["Round"] + ")"
# Transpose the table
transposed_table_df = transposed_df[table_columns + ["Supplier_Round"]].set_index("Supplier_Round").transpose().reset_index().rename(columns={"index": "Field"})
# Sort columns to place same supplier's Round 1 and Round 2 side by side
supplier_round_cols = [col for col in transposed_table_df.columns if col != "Field"]
sorted_cols = []
suppliers = sorted(set(transposed_df["Supplier"]))
for supplier in suppliers:
    for round_name in ["Round 1", "Round 2"]:
        col_name = f"{supplier} ({round_name})"
        if col_name in supplier_round_cols:
            sorted_cols.append(col_name)
transposed_table_df = transposed_table_df[["Field"] + sorted_cols]

# Display transposed Supplier Cost Comparison table
st.subheader("Transposed Supplier Cost Comparison (Round 1 and Round 2)")
if not transposed_table_df.empty:
    st.dataframe(transposed_table_df)
else:
    st.write("No data matches the selected filters. Please adjust the filters to see the transposed comparison table.")

# Download Excel file for transposed table
transposed_excel_data = generate_transposed_excel(transposed_table_df)
st.download_button(
    label="Download Transposed Table as Excel",
    data=transposed_excel_data,
    file_name="transposed_supplier_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Visualizations
st.subheader("Visualizations")
if not filtered_df.empty:
    sorted_df = filtered_df.sort_values(by="Total Budget", ascending=False)
    fig_budget = px.bar(sorted_df, x="Supplier", y="Total Budget", color="Round", title="Total Budget by Supplier (Highest to Lowest)",
                        labels={"Total Budget": "Total Budget (RS)"}, barmode="group")
    st.plotly_chart(fig_budget, use_container_width=True)

    material_cols = [col for col in filtered_df.columns if col not in ["Supplier", "Round", "Total Budget"]]
    if "All" not in selected_materials:
        material_cols = [col for col in material_cols if col in selected_materials]
    if material_cols:
        fig_materials = go.Figure()
        for material in material_cols:
            fig_materials.add_trace(go.Bar(x=filtered_df["Supplier"], y=filtered_df[material], name=material, marker_color=px.colors.qualitative.Plotly[material_cols.index(material) % len(px.colors.qualitative.Plotly)]))
        fig_materials.update_layout(barmode="group", title="Material Costs by Supplier", xaxis_title="Supplier", yaxis_title="Cost (RS)")
        st.plotly_chart(fig_materials, use_container_width=True)

# AI-based recommendation
st.subheader("AI-Based Supplier Recommendation")
if not filtered_df.empty:
    if round_filter == "All":
        best_supplier_r1, score_r1, round_r1, _ = recommend_best_supplier(df, "Round 1") if not df[df["Round"] == "Round 1"].empty else (None, 0, None, None)
        best_supplier_r2, score_r2, round_r2, _ = recommend_best_supplier(df, "Round 2") if not df[df["Round"] == "Round 2"].empty else (None, 0, None, None)
        if best_supplier_r1 and best_supplier_r2:
            st.write(f"Round 1 Recommendation: {best_supplier_r1} (Score: {score_r1:.2f})")
            st.write(f"Round 2 Recommendation: {best_supplier_r2} (Score: {score_r2:.2f})")
            if score_r1 > score_r2:
                st.write(f"Overall Best Supplier: {best_supplier_r1} from Round 1 with a score of {score_r1:.2f}")
            else:
                st.write(f"Overall Best Supplier: {best_supplier_r2} from Round 2 with a score of {score_r2:.2f}")
        elif best_supplier_r1:
            st.write(f"Best Supplier: {best_supplier_r1} from Round 1 (Score: {score_r1:.2f})")
        elif best_supplier_r2:
            st.write(f"Best Supplier: {best_supplier_r2} from Round 2 (Score: {score_r2:.2f})")
    else:
        best_supplier, score, round, _ = recommend_best_supplier(df, round_filter)
        if best_supplier:
            st.write(f"Best Supplier for {round_filter}: {best_supplier} (Score: {score:.2f})")
        else:
            st.write(f"No valid suppliers available for {round_filter}.")
    # Display Round 1 vs Round 2 comparison
    st.write(f"Round Comparison: {compare_rounds(df)}")
else:
    st.write("No data available for recommendations. Please adjust filters or upload more documents.")

# Question answering
st.subheader("Ask Questions About the Invoices")
if st.session_state.get("Round 1_data") or st.session_state.get("Round 2_data"):
    question_round = st.selectbox("Select Round for Question", ["All", "Round 1", "Round 2"])
    user_question = st.text_input("Ask a question about the invoices (e.g., 'What is the cost of tiles for Astle Empires?' or 'What is the total budget for Round 2?'):")
    if user_question:
        with st.spinner("💡 Thinking..."):
            answer = ask_question_about_invoice(user_question, question_round if question_round != "All" else None)
        st.success("Answer:")
        st.write(answer)
else:
    st.info("No invoices available to query. Please upload supplier documents.")