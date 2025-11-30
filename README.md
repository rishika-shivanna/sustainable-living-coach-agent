Sustainable Living Coach Agent
AI-Powered Multi-Agent System for Carbon Footprint Insights from Shopping Receipts

Built for Kaggle’s AI Agents Intensive Capstone
Overview

The Sustainable Living Coach Agent is a multi-agent pipeline that analyzes shopping receipts and provides:

Carbon footprint estimation

Eco-friendly recommendations (LLM-powered or offline fallback)

Daily sustainability scoring

Progress tracking over time

Beautiful, structured output for users

This project is designed according to Kaggle’s ADK-style agent structure, but works fully in a standalone Python environment or inside a Kaggle Notebook.
Features
1. Receipt Extraction (ReceiptExtractorAgent)

Parses raw text or .txt receipts

Extracts store, date, item names, quantities, prices

Robust error handling for messy supermarket text

2. Carbon Footprint Mapping (CarbonCalculatorAgent)

Categorizes items (meat, dairy, snacks, electronics, etc.)

Assigns an educational-level CO₂ impact value

Computes per-item & total emission in kg CO₂
3. Personalized Recommendations (RecommendationAgent / LLM)

Two modes:

LLM (Gemini or ChatGPT) for natural smart suggestions

Offline fallback if API not available

Suggestions based on high-emission categories

4. Sustainability Progress Tracking (ProgressTrackerAgent)

Saves carbon logs in sustainability_history.json

Computes:

Average daily footprint

Total days logged

Sustainability score (0–100)

Trend insight for long-term behavior change

5. Notebook-Friendly Display

Beautiful output formatting with:

Unicode icons

Tables

Blocks

Emoji score bars

Clean and user-friendly for Kaggle submission reviewers

sustainable-living-coach-agent/
│
├── agent.py                    # Full multi-agent implementation
├── Sustainable_Living_Coach.ipynb
├── requirements.txt            # Dependencies
├── README.md                   # You're reading this :)
└── architecture.png            # (optional)
