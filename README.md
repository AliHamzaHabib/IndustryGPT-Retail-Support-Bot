# IndustryGPT: Specialized LLM Bot for Retail and E-Commerce Support

![Project Type](https://img.shields.io/badge/Project-FineTuned%20LLM%20Capstone-blue)
![Model](https://img.shields.io/badge/Model-TinyLlama--1.1B--Chat-orange)
![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Hugging%20Face-green)
![Method](https://img.shields.io/badge/Method-QLoRA%20%284--bit%20Quantization%29-purple)

An industry-specific fine-tuned Large Language Model (LLM) designed to automate and enhance customer support workflows for retail and e-commerce platforms.

---

## 📌 Project Overview
Customer service teams face high volumes of repetitive inquiries regarding order tracking, product returns, shipping updates, and cart management. This project builds a specialized generative customer-support chatbot by fine-tuning **TinyLlama-1.1B-Chat** using **QLoRA (Quantized Low-Rank Adaptation)** on a comprehensive public retail dataset. 

The bot is engineered to deliver policy-aligned, professional, and empathetic guidance across various e-commerce support scenarios.

---

## 🎯 Problem Statement
Standard pre-trained foundational models often lack the specific domain language, tone, and procedural familiarity required for efficient retail support. This capstone explores whether a compact, open-source small language model (SLM) can be adapted to master e-commerce support interactions while maintaining high efficiency and low computational overhead.

---

## ⚙️ Technical Architecture & Workflow
1. **Data Ingestion & Cleaning:** Leveraged the public *Bitext Retail and E-Commerce LLM Chatbot Training Dataset* (44,884 raw records), applying rigorous normalization, PII filtering, and deduplication down to 23,795 clean records.
2. **Exploratory Data Analysis (EDA):** Analyzed data distributions across 13 retail categories and 46 customer support intents, evaluating text length percentiles to establish optimal training sequence constraints (max length: 256 tokens).
3. **Model Selection & Quantization:** Loaded **TinyLlama-1.1B-Chat-v1.0** with 4-bit NormalFloat (`NF4`) quantization via `bitsandbytes` to operate smoothly within a standard Google Colab T4 GPU memory budget (~15 GB VRAM).
4. **Parameter-Efficient Fine-Tuning (PEFT):** Configured LoRA adapters (`r=16`, `lora_alpha=32`) targeting attention projection modules (`q_proj`, `v_proj`, `k_proj`, `o_proj`) to train a fraction of the model parameters efficiently.
5. **Robust Inference Pipeline:** Implemented prompt consistency checks and structured generation controls to ensure clean, multi-line, professional customer support answers.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch, Hugging Face `transformers`, `datasets`, `peft`, `accelerate`
* **Quantization & Optimization:** `bitsandbytes` (4-bit NF4)
* **Data Visualization & Analysis:** Matplotlib, Seaborn, Pandas, NumPy

---

## 📂 Repository Structure
```text
IndustryGPT-Retail-Support-Bot/
│
├── notebooks/
│   └── IndustryGPT_Retail_Ecommerce_Support_Bot.ipynb   # Complete, runnable Google Colab notebook
├── docs/
│   └── Capstone_Project_Report_IndustryGPT_documentation.docx  # Official project report and documentation
└── README.md

## ⚠️ Limitations
* **Lack of Real-Time Backend Integration:** The fine-tuned model provides accurate guidance based on its training data, but it cannot connect to live retailer databases to check real-time order statuses, modify active customer accounts, or process live financial transactions without external API coupling or RAG systems.
* **Compact Model Scale:** Built on a 1.1B parameter architecture, which balances efficiency on local T4 GPUs but offers shallower multi-step reasoning compared to larger 7B+ models.
* **Stateless Execution:** Operates on a single instruction-response evaluation structure without advanced multi-turn conversation memory management.

## 🚀 How to Run the Notebook
1. Open the notebook `IndustryGPT_Retail_Ecommerce_Support_Bot.ipynb` inside Google Colab.
2. Ensure your runtime is set to a T4 GPU (`Runtime > Change runtime type > T4 GPU`).
3. Run all cells sequentially to execute data preprocessing, EDA charts, QLoRA fine-tuning, and inference testing.

## 👤 Author
* **Ali Hamza Habib**
* Capstone Project submitted as part of the Data Science & AI Engineering Curriculum.
