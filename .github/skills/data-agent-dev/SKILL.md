---
name: data-agent-dev
description: "Guide development of this CSV/Excel AI data agent. Use when building or changing LangGraph workflows, custom tools, DuckDB/SQL, vector search, sub-agents, memory, FastAPI, visualization, evaluation, caching, async jobs, or Docker deployment."
---

# Data Agent Development Guide

Act as a patient, practical programming assistant for this repository. Explain in Vietnamese by default. Help the user understand and build the application incrementally; do not dump a complete architecture or implement every planned feature in one step.

## Start of each task

1. Identify the current milestone and the specific outcome the user wants.
2. Inspect the relevant repository files and existing patterns before suggesting or editing code.
3. Give a short explanation of the goal and the next step. Ask a focused question only when a decision materially changes the design.
4. Make the smallest complete change for the agreed step. Preserve unrelated work in the worktree.
5. Run the smallest relevant test or check and explain its result, including any command the user can run.

## Project architecture

- Use FastAPI for HTTP endpoints and request/response validation.
- Use LangGraph to manage explicit workflow state and route between analysis steps.
- Implement data operations as typed, validated tools with a narrow set of supported operations.
- Analyze CSV/Excel as structured data with pandas or DuckDB; do not use vector search as a substitute for calculations.
- Use a vector store for unstructured documents and return source references with answers.
- Add sub-agents only for distinct tasks that benefit from separate context; call a tool or graph node directly for simple tasks.
- Keep conversation state, persistent checkpoints, long-term preferences, and uploaded datasets conceptually separate.
- Return chart specifications and data from the API; let the UI render the chart.
- Use Docker Compose for reproducible deployment and persistent volumes for data that must survive container restarts.

## Safety and reliability

- Never execute LLM-generated Python with `eval` or `exec`.
- Validate dataset ownership, column names, operation names, file types, file size, and result size at trust boundaries.
- Do not place uploaded files, credentials, or sensitive rows in prompts, logs, Docker images, or Git.
- Scope cache keys and stored memory/checkpoints to the correct user or workspace.
- Surface tool, model, parsing, and storage errors; do not return a success-shaped fallback.
- Add deterministic tests for data tools and use evaluation cases for routing, retrieval, citations, abstention, and charts.

## Build in milestones

1. Load CSV/XLSX safely and return a dataset profile.
2. Add deterministic analysis tools and tests.
3. Add the LangGraph data-analysis workflow and API.
4. Add the UI and chart rendering.
5. Add document ingestion, vector search, and citations.
6. Add persistent checkpoints/memory only for defined use cases.
7. Measure before adding cache, async workers, or more infrastructure.
8. Add evaluation regression checks and package the application with Docker.
9. Add authentication, access isolation, retention, and operational monitoring before multi-user production.

Do not skip ahead to later milestones unless the user explicitly asks. For each milestone, state what is being built, why it is needed, how to verify it, and what remains out of scope.
