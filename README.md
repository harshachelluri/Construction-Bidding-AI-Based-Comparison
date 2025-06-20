# 🏗️ Construction Bidding AI-Based Comparison

### 📊 Intelligent Supplier Evaluation Tool for Construction Projects

This project streamlines the comparison of supplier bids using AI-powered analysis. Upload documents from multiple rounds of bidding, extract costs, visualize trends, and receive automated supplier recommendations — all within a user-friendly dashboard.

> 💡 *"Make smarter, faster, and more informed construction bidding decisions using AI."*

---

## 📌 Problem Statement

In construction projects, comparing and evaluating quotations from multiple suppliers across multiple bidding rounds is time-consuming and prone to human error. There’s often a need to:

- Parse data from various formats (PDFs, DOCX, images, etc.)
- Extract and standardize supplier and material cost information
- Analyze and visualize material cost differences across bidding rounds
- Automatically recommend the best supplier based on material coverage and budget
- Use AI to answer queries based on the uploaded data

---

## 🧪 Problem Case

Imagine a real-estate developer receives supplier bids from 10+ vendors for the foundation, plumbing, tiling, and woodwork for a new housing project. The bids arrive in multiple formats: PDFs, scanned images, Word docs, and spreadsheets. Each supplier uses a different layout and format, and includes costs in varying styles (e.g., "Rs 20,000", "INR 20000").

Manually comparing these invoices to identify which vendor provides the best value — in terms of cost and material coverage — would take several hours or days.

**This dashboard solves that by:**

- Extracting all relevant cost data from diverse document types automatically
- Standardizing and visualizing that information in clean tables and charts
- Allowing AI-driven filtering and question-answering
- Recommending the supplier with the best balance of cost and material variety

---

## 🎯 Solution

This app addresses the above problems by:

- ✅ Supporting file uploads in `.docx`, `.pdf`, `.csv`, `.jpg`, `.png`
- ✅ Performing OCR to extract text from scanned or image-based invoices
- ✅ Extracting structured data (supplier name, material costs, total budget)
- ✅ Displaying cost comparison tables (original and transposed by round)
- ✅ Providing AI-based recommendations based on coverage and budget efficiency
- ✅ Enabling question-answering on invoice data using Gemini AI
- ✅ Exporting final bidding comparison as an Excel file for sharing/reporting

---

## 🚀 Features

- 📄 Upload and parse multi-format supplier invoices  
- 📊 Visualize cost comparisons and material breakdowns  
- 🧠 AI-powered supplier recommendations (coverage vs budget)  
- ❓ Natural language queries using Gemini AI (e.g., "Who is the cheapest supplier for tiles?")  
- 📤 Export final bidding analysis to Excel in original and transposed format  
- 🧹 Filter by round, material type, and supplier

---

## 🧰 Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **Plotly**
- **Tesseract OCR + pdf2image**
- **PyPDF2**
- **python-docx**
- **Google Gemini AI (Generative Model)**

---

## 📁 File Upload Support

Supported input formats:

- ✅ PDF (machine-readable and scanned)
- ✅ DOCX (Word documents)
- ✅ CSV (structured data)
- ✅ PNG, JPG, JPEG (scanned or photographed invoices)

---

## 📤 Output

All analyzed and processed bidding data can be exported to:

- 📁 **Excel files** (`.xlsx`)
  - Original comparison table (rows = suppliers, columns = materials)
  - Transposed table (columns = supplier & round; rows = materials)

Ideal for reporting, audits, or final decision-making documentation.

---

## 📸 Screenshots
![Image](https://github.com/user-attachments/assets/30949106-716d-4453-a9d5-cd4d627ac454)


