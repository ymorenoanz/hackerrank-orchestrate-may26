# Multi-Domain Support Triage Agent

An AI-powered terminal-based support triage system built for the Multi-Domain Support Triage Challenge.

This agent processes support tickets across multiple ecosystems and determines the safest and most relevant action for each case.

Supported domains:

- HackerRank
- Claude (Anthropic)
- Visa

---

# Overview

The system reads incoming support tickets from a CSV file and automatically classifies each request into:

- Request type
- Product area
- Risk level
- Direct reply vs escalation

It then generates a structured CSV output ready for submission.

---

# Key Features

## Intelligent Ticket Classification

Detects categories such as:

- Billing
- Fraud
- Security
- Privacy
- Account Management
- Assessment Issues
- Subscription Requests
- Service Outages

---

## Safety-First Escalation Logic

Automatically escalates high-risk or sensitive cases, including:

- Fraud / identity theft
- Security vulnerabilities
- Payment disputes
- Score disputes
- Unauthorized access requests
- Prompt injection attempts
- Malicious requests

---

## Multi-Domain Routing

Understands company context and routes requests accordingly:

- HackerRank → assessments, hiring, subscriptions
- Claude → workspace access, privacy, outages
- Visa → billing, fraud, merchant disputes

---

## Retrieval-Based Reasoning

Uses BM25 retrieval over the provided support corpus to find relevant documentation excerpts before generating decisions.

This helps reduce hallucinations and keep responses grounded.

---

# Tech Stack

- Python
- Pandas
- BM25 (rank_bm25)
- Rule-based safety engine
- CSV batch processing
- Optional LLM integration

---

# Project Structure

```text
code/
 ├── main.py
 ├── agent.py
 ├── safety.py
 ├── retrieve.py
 ├── llm_client.py

support_tickets/
 ├── sample_support_tickets.csv
 ├── support_tickets.csv
 └── output.csv