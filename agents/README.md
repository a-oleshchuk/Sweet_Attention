# Agents

This folder contains the support-agent implementation.

Current scope:
- `support_agent/`: the LangGraph implementation for the supervised support agent

Run a single support-agent turn with:

```powershell
..\.venv\Scripts\python -m agents.support_agent.cli --conversation-id demo-1 --user-name "Jane Miller" --message "I paid but still do not have access."
```
