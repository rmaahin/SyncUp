"""State reducers for LangGraph.

Re-exports ``operator.add`` as ``append_reducer`` for explicit usage in
annotated state fields.  Custom reducers (deduplication, validation-on-merge)
will be added here as needed.
"""

import operator

append_reducer = operator.add

__all__ = ["append_reducer"]
