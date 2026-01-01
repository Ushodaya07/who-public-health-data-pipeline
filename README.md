# 🌍 WHO Public Health Data Pipeline (Kedro)

![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)
![Kedro](https://img.shields.io/badge/Kedro-0.19+-purple.svg)
![Data Engineering](https://img.shields.io/badge/Data-Engineering-blue.svg)

An end-to-end **Data Engineering pipeline** built using **Kedro** to ingest, clean, transform, and curate **WHO public health and outbreak datasets** for downstream analytics, reporting, and data science use cases.

---

## 📌 Project Overview

This project implements a **production-grade data pipeline** for handling **global public health data** published by the **World Health Organization (WHO)**.  
The pipeline follows **best practices in Data Engineering**, including modular design, configuration-driven workflows, reproducibility, and clear data lineage.

### Key Objectives
- Automate ingestion of raw WHO datasets  
- Clean & standardize health and outbreak data  
- Generate curated datasets for analytics & ML  
- Ensure reproducibility using Kedro pipelines  
- Prepare data for reporting and dashboards  

---

## 🧠 Pipeline Architecture

The pipeline follows a layered data architecture:

1. **Raw Layer**  
   - Ingests raw WHO datasets (CSV / Excel / API extracts)

2. **Processing Layer**  
   - Data cleaning (null handling, type casting)
   - Standardization of country codes & dates
   - Feature generation and aggregations

3. **Analytics & Reporting Layer**  
   - Curated datasets for analysis
   - Data science–ready tables
   - Reporting outputs

4. **Orchestration**  
   - Managed using **Kedro pipelines**
   - Configuration-driven via YAML

---

## 📁 Project Folder Structure

```text
who-public-health-data-pipeline/
│
├── who-outbreak-pipeline/
│   ├── conf/
│   │   ├── base/
│   │   │   ├── catalog.yml
│   │   │   ├── parameters.yml
│   │   │   ├── parameters_data_processing.yml
│   │   │   ├── parameters_data_science.yml
│   │   │   └── parameters_reporting.yml
│   │   └── logging.yml
│   │
│   ├── data/
│   │   ├── 01_raw/
│   │   ├── 02_intermediate/
│   │   ├── 03_primary/
│   │   ├── 04_feature/
│   │   ├── 05_model_input/
│   │   └── 07_reporting/
│   │
│   ├── notebooks/
│   │
│   ├── src/who_outbreak_pipeline/
│   │   ├── pipelines/
│   │   │   ├── who_data/
│   │   │   ├── data_processing/
│   │   │   ├── data_science/
│   │   │   └── reporting/
│   │   ├── pipeline_registry.py
│   │   └── settings.py
│   │
│   ├── streamlit_app/
│   │   └── dashboard.py
│   │
│   └── pyproject.toml
│
├── .gitignore
└── README.md
```
---
## 🔧 Tech Stack

- **Language:** Python 3.9+
- **Framework:** Kedro (Pipeline orchestration)
- **Data Processing:** Pandas, NumPy
- **Configuration:** YAML
- **Visualization (Optional):** Streamlit
- **Version Control:** Git & GitHub

---

## 🧩 Kedro Pipelines

### WHO Data Pipeline
- Raw data ingestion
- Cleaning & normalization
- Feature engineering
- Aggregations by country, region, and date

### Data Processing Pipeline
- Schema validation
- Missing value handling
- Consistent formatting

### Data Science Pipeline
- Feature preparation
- Model-ready datasets

### Reporting Pipeline
- Final curated datasets
- Outputs for dashboards and BI tools

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Ushodaya07/who-public-health-data-pipeline.git
cd who-public-health-data-pipeline/who-outbreak-pipeline
```
### 2️⃣ Create Virtual Environment
```bash
python -m venv .venv
source .venv/Scripts/activate
```
### 3️⃣ Install Dependencies
```bash
pip install -e .
```
### 4️⃣ Run Kedro Pipeline
```bash
kedro run
```
### Run a specific pipeline:
```bash
kedro run --pipeline who_data
```
---
## 📊 Outputs

- Cleaned & standardized public health datasets  
- Feature-engineered tables for analytics  
- Reporting-ready data layers  
- Reproducible pipelines with clear lineage  

---

## 🎯 Future Enhancements

- Integrate WHO live APIs  
- Add data quality checks (Great Expectations)  
- Orchestrate with Airflow / Prefect  
- Store outputs in Cloud (S3 / Azure Blob / GCS)  
- Add BI dashboards (Power BI / Tableau)  

---

## 👨‍💻 Author

**Dasari Ushodaya**  
GitHub: https://github.com/Ushodaya07  
