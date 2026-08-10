# 🤖 Enterprise Multi-Agent AI Customer Support Platform

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20100%25-brightgreen.svg)](https://docs.pytest.org/)

An enterprise-grade, asynchronous AI customer support platform leveraging **Dynamic Multi-Agent Orchestration**, **Hybrid RAG (ChromaDB + BM25)**, and stateful session memory. Built with high performance, scalability, and modularity in mind.

## 🚀 Live Demo

- 🌐 **Live Application:** https://ai-customer-agent-02vo.onrender.com/
- 📚 **API Documentation:** https://ai-customer-agent-02vo.onrender.com/docs

> Deployed on Render with a Streamlit interface and FastAPI REST API.

---

## 🏛️ System Architecture

```text
                  +-----------------------------------+
                  |   Streamlit Interactive UI        |
                  |   (Session State & Document Upload) |
                  +-----------------+-----------------+
                                    |
                                    v (HTTP REST API)
                  +-----------------+-----------------+
                  |      FastAPI API Gateway          |
                  |     (/chat, /upload, /health)     |
                  +-----------------+-----------------+
                                    |
                                    v
                  +-----------------+-----------------+
                  |   Agent Orchestrator (Groq)       |
                  |   (Intent Router & Personas)      |
                  +--------+----------------+---------+
                           |                |
           +---------------+                +---------------+
           |                                                |
           v                                                v
+----------+----------+                          +----------+----------+
|  Policy Specialist  |                          | Technical Specialist|
+----------+----------+                          +----------+----------+
           |                                                |
           +-----------------------+------------------------+
                                   |
                                   v
                   +---------------+---------------+
                   |   Hybrid Retrieval Engine     |
                   |  (Dense Vector + Sparse BM25) |
                   +---------------+---------------+
                                   |
                   +---------------+---------------+
                   |  ChromaDB + In-Memory Store   |
                   +-------------------------------+
