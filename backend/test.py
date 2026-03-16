from src.graph.main import graph, _NODE_FUNCTIONS

# Confirm the real agent is wired in
print(_NODE_FUNCTIONS["task_decomposition"].__module__)
# → "agents.task_decomposition"  (NOT "graph.main")

# The full graph currently ends immediately at supervisor
# (supervisor_router returns "__end__" by default), so a full
# graph.invoke() won't reach task_decomposition yet.
# That's expected — supervisor routing is still a stub.