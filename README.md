# SAHRAchat: Compound AI Medical Triage Assistant

SAHRAchat (Sexual and Reproductive Health Assistant) is a proof-of-concept clinical triage and educational chatbot. Built to run on **Red Hat OpenShift AI (RHOAI)**, this project demonstrates a "Compound AI System" that enforces deterministic clinical safety guardrails over a fine-tuned generative AI model.

## Overview

In clinical environments, Large Language Models (LLMs) require strict grounding to prevent medical hallucinations and ensure compliance with established guidelines. SAHRAchat achieves this by separating the conversational interface from the clinical decision-making logic:

1. **Inference & Persona (vLLM + LoRA):** The system utilizes `Mistral-Small-24B-Instruct` as the base model. To achieve clinical empathy and proper bedside manner, a low-rank adapter (LoRA) was fine-tuned on nursing transcripts. This adapter is dynamically loaded at runtime using vLLM (`--enable-lora=True`), allowing for rapid persona swapping without deploying multiple heavyweight models.
2. **Deterministic State Routing (LangGraph):** A Python-based state machine controls the conversational flow. The LLM is restricted from generating recommendations until a strict set of patient variables (Preferences, Age, Blood Pressure, Clotting History, etc.) are successfully extracted and verified in the session state.
3. **Structured RAG (Docling + PGVector):** The 2024 CDC Medical Eligibility Criteria (MEC) tables were programmatically parsed using IBM Docling and stored as explicit semantic rules in a CloudNativePG (PostgreSQL + pgvector) database. The LangGraph router forces the LLM to filter recommendations strictly against these retrieved CDC rules.

## System Architecture

![SAHRAchat RHOAI Architecture](assets/SAHRAchat_arch_diagram.png)

## State Machine Workflow

The LangGraph application enforces a multi-stage triage process. The model cannot bypass the health screening stages to offer a consultation.

![LangGraph State Flow](assets/SAHRAchat_convo_flow_diagram.png)

## Repository Structure

```text
.
├── app/                        # Main LangGraph application
│   ├── main.py                 # CLI entry point for the chat loop
│   ├── graph.py                # LangGraph edge/node definitions
│   ├── nodes.py                # LLM invocations and system prompts
│   ├── state.py                # TypedDict for session state variables
│   ├── stage_3_subgraph.py     # CDC health risk extraction logic
│   └── config.py               # Environment and LLM client configuration
├── data/                       # Knowledge base and training assets
│   ├── cdc_mec_tables_only.pdf # Source CDC guidelines
├── fine-tuning/                # Model training scripts
│   └── train_lora.py           # Unsloth/TRL script for Mistral 24B fine-tuning
├── infrastructure/             # Deployment and data ingestion scripts
│   ├── ingest_cdc.py           # Parses PDF via Docling and loads PGVector
│   └── upload_models.py        # Syncs base models and LoRA adapters to MinIO S3
├── assets/                     # Architecture and workflow diagrams
├── requirements-app.txt        # Runtime dependencies (LangGraph, vLLM client)
└── requirements-lora.txt       # Training dependencies (Unsloth, PyTorch)

## 🖥️ Running the Clinical MVP GUI (Gradio)

While the core LangGraph logic can be tested in the terminal using `main.py`, we have included a fully featured Gradio Web UI designed specifically for clinical and architectural demonstrations.

The GUI features a "Split-Screen" design to prove the auditable nature of our architecture:
* **Left Pane (The Patient View):** A clean chat interface interacting with our LoRA fine-tuned Mistral model.
* **Right Pane (The Architect View):** A live JSON state tracker showing LangGraph strictly extracting clinical variables, plus an expandable module showing the exact 2024 CDC MEC safety rules retrieved via pgvector (RAG).

### How to Launch the GUI
Ensure your virtual environment is active and your OpenShift vLLM instance is running.

1. Navigate to the root of the project directory.
2. Run the Gradio app:
   ```bash
   python app/app_gradio.py
3. Look at the terminal output for the public URL (e.g., https://xxxxxx.gradio.live). 
    Click this link to open the UI in a full-screen, unblocked browser tab.

Note: The Gradio UI includes a share=True flag by default. This safely tunnels the UI through the Jupyter/OpenShift proxy so it can be easily shared during live presentations.