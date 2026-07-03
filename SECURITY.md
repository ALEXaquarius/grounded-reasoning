# Security Policy

## Reporting a vulnerability

Please report security issues privately via **GitHub Security Advisories**
on this repository (Security tab → "Report a vulnerability") rather than
opening a public issue. We aim to acknowledge reports within a few days.

## Secrets and API keys

- This project **never** hardcodes API keys. All LLM providers read their key
  from environment variables (e.g. `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`).
- `.env` is git-ignored. Do **not** commit `.env` or any credential.
- If a key is ever exposed (e.g. pasted into an issue, log, or commit), treat it
  as compromised and **rotate it immediately** in the provider dashboard.

## Scope

The core verifier runs locally with no network access and no model tokens. The
optional experiment scripts under `src/experiments/` call external LLM APIs only
when you provide a key; they are not required to use the library.
