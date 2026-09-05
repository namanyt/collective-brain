# Global Agent Directives: Collective Brain & Memory Protocol

You are connected to the **Collective Brain MCP Server** (running over SSE on `http://localhost:5006/sse` with FastEmbed embeddings and Cross-Encoder reranking).

## Mandatory Memory Protocol for All Tasks:

1. **Active Context Recall**:
   - At the beginning of any project, task, or ambiguous user query, check for relevant knowledge using `recall_memories` or `unified_search`.
   - Recall user habits, technical preferences, server configuration details, or prior project decisions.

2. **Continuous Memory Persistence**:
   - Whenever the user expresses a preference (e.g. library choices, response styles, workflow patterns), immediately record it using `remember_fact_or_preference(fact=..., category="preference", tags=[...])`.
   - Whenever an important architecture decision, hardware constraint, or environment detail is discovered, record it using `remember_fact_or_preference(fact=..., category="fact"|"rule", tags=[...])` or `save_note(title=..., content=..., tags=[...])`.
   - When concluding a major task, persist a concise summary note using `save_note` so future sessions and other agents retain full continuity.

3. **Knowledge Base Documentation**:
   - When encountering long manuals, API guides, or architecture documents, index them using `index_document` rather than letting them get lost when context trims.

4. **Living Consciousness Maintenance**:
   - Keep the collective memory clean and accurate: update changed facts with `update_memory` and remove obsolete entries with `forget_memory`.
