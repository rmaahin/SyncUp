"""Tests for the LLM factory and Task Decomposition agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from state.schema import SyncUpState, StudentProfile, TaskStatus, UrgencyLevel

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_TASKS_JSON: str = json.dumps(
    {
        "tasks": [
            {
                "id": "task-setup-repo",
                "title": "Set up repository",
                "description": "Initialize Git repo with project structure and CI",
                "effort_hours": 3.0,
                "required_skills": ["git", "python"],
                "urgency": "high",
                "dependencies": [],
            },
            {
                "id": "task-design-db",
                "title": "Design database schema",
                "description": "Create ERD and write SQL migrations",
                "effort_hours": 8.0,
                "required_skills": ["sql", "python"],
                "urgency": "critical",
                "dependencies": ["task-setup-repo"],
            },
            {
                "id": "task-build-api",
                "title": "Build REST API endpoints",
                "description": "Implement FastAPI routes for CRUD operations",
                "effort_hours": 12.0,
                "required_skills": ["python", "fastapi"],
                "urgency": "high",
                "dependencies": ["task-design-db"],
            },
            {
                "id": "task-frontend-ui",
                "title": "Build frontend UI",
                "description": "Create React components for the student portal",
                "effort_hours": 15.0,
                "required_skills": ["react", "typescript"],
                "urgency": "medium",
                "dependencies": ["task-build-api"],
            },
            {
                "id": "task-write-docs",
                "title": "Write documentation",
                "description": "Write user guides and API documentation",
                "effort_hours": 4.0,
                "required_skills": ["writing"],
                "urgency": "low",
                "dependencies": [],
            },
        ]
    }
)

SAMPLE_BRIEF: str = (
    "Build a web-based student portal where students can view grades, "
    "submit assignments, and communicate with professors."
)


def _make_mock_llm(responses: list[str]) -> MagicMock:
    """Create a mock LLM that returns the given responses in order.

    Args:
        responses: List of content strings the mock should return on
            successive ``.invoke()`` calls.

    Returns:
        A ``MagicMock`` whose ``.invoke()`` returns mock AIMessage objects.
    """
    llm = MagicMock()
    side_effects: list[MagicMock] = []
    for text in responses:
        msg = MagicMock()
        msg.content = text
        side_effects.append(msg)
    llm.invoke.side_effect = side_effects
    return llm


def _make_state(
    project_brief: str = "",
    student_profiles: list[StudentProfile] | None = None,
) -> SyncUpState:
    """Create a minimal ``SyncUpState`` for testing."""
    return SyncUpState(
        project_id="test-project",
        project_brief=project_brief,
        student_profiles=student_profiles or [],
    )


# =========================================================================
# LLM Factory tests
# =========================================================================


class TestLLMFactory:
    """Tests for ``backend/src/llm/__init__.py``."""

    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-key-123",
            "LLM_HIGH_TIER_MODEL": "llama-3.3-70b-versatile",
        },
        clear=False,
    )
    def test_get_high_tier_returns_chat_groq(self) -> None:
        from llm import get_high_tier_llm

        llm = get_high_tier_llm()
        assert llm.model_name == "llama-3.3-70b-versatile"
        assert llm.temperature == 0.7

    @patch.dict(
        "os.environ",
        {
            "GROQ_API_KEY": "test-key-123",
            "LLM_LOW_TIER_MODEL": "llama-3.1-8b-instant",
        },
        clear=False,
    )
    def test_get_low_tier_returns_chat_groq(self) -> None:
        from llm import get_low_tier_llm

        llm = get_low_tier_llm()
        assert llm.model_name == "llama-3.1-8b-instant"
        assert llm.temperature == 0.7

    @patch.dict(
        "os.environ",
        {"GROQ_API_KEY": "", "LLM_HIGH_TIER_MODEL": "llama-3.3-70b-versatile"},
        clear=False,
    )
    def test_missing_api_key_raises_runtime_error(self) -> None:
        from llm import get_high_tier_llm

        with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
            get_high_tier_llm()

    def test_default_model_names(self) -> None:
        from llm import get_high_tier_llm, get_low_tier_llm

        import os

        # Build env without model name vars to test defaults
        env = {k: v for k, v in os.environ.items() if k not in (
            "LLM_HIGH_TIER_MODEL", "LLM_LOW_TIER_MODEL"
        )}
        env["GROQ_API_KEY"] = "test-key-123"

        with patch.dict("os.environ", env, clear=True):
            high = get_high_tier_llm()
            low = get_low_tier_llm()
            assert high.model_name == "llama-3.3-70b-versatile"
            assert low.model_name == "llama-3.1-8b-instant"

    @patch.dict("os.environ", {"GROQ_API_KEY": "test-key-123"}, clear=False)
    def test_custom_temperature(self) -> None:
        from llm import get_high_tier_llm

        llm = get_high_tier_llm(temperature=0.0)
        # ChatGroq clamps 0.0 to 1e-08 internally
        assert llm.temperature <= 0.01

    @patch.dict(
        "os.environ",
        {"GROQ_API_KEY": "test-key-123", "LLM_HIGH_TIER_MODEL": "custom-model"},
        clear=False,
    )
    def test_custom_model_from_env(self) -> None:
        from llm import get_high_tier_llm

        llm = get_high_tier_llm()
        assert llm.model_name == "custom-model"


# =========================================================================
# Task Decomposition — empty brief
# =========================================================================


class TestTaskDecompositionEmptyBrief:
    """Edge cases: empty or whitespace-only project_brief."""

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_empty_string_returns_empty(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        state = _make_state(project_brief="")
        result = task_decomposition(state)

        assert result["task_array"] == []
        assert result["dependency_graph"] == {}
        mock_factory.assert_not_called()

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_whitespace_returns_empty(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        state = _make_state(project_brief="   \n\t  ")
        result = task_decomposition(state)

        assert result["task_array"] == []
        assert result["dependency_graph"] == {}
        mock_factory.assert_not_called()


# =========================================================================
# Task Decomposition — happy path
# =========================================================================


class TestTaskDecompositionHappyPath:
    """Valid project brief with mocked LLM returning valid JSON."""

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_valid_brief_produces_tasks(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5
        assert result["task_array"][0].id == "task-setup-repo"
        assert result["task_array"][0].title == "Set up repository"
        mock_llm.invoke.assert_called_once()

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_dependency_graph_built(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        dep_graph: dict[str, list[str]] = result["dependency_graph"]
        assert dep_graph["task-setup-repo"] == []
        assert dep_graph["task-design-db"] == ["task-setup-repo"]
        assert dep_graph["task-build-api"] == ["task-design-db"]
        assert dep_graph["task-frontend-ui"] == ["task-build-api"]
        assert dep_graph["task-write-docs"] == []

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_urgency_enum_mapping(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        tasks = result["task_array"]
        urgency_map = {t.id: t.urgency for t in tasks}
        assert urgency_map["task-setup-repo"] == UrgencyLevel.HIGH
        assert urgency_map["task-design-db"] == UrgencyLevel.CRITICAL
        assert urgency_map["task-frontend-ui"] == UrgencyLevel.MEDIUM
        assert urgency_map["task-write-docs"] == UrgencyLevel.LOW

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_all_tasks_have_todo_status(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        for task in result["task_array"]:
            assert task.status == TaskStatus.TODO
            assert task.assigned_to is None
            assert task.deadline is None

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_skills_from_profiles_in_prompt(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        profiles = [
            StudentProfile(
                student_id="s1",
                name="Alice",
                email="alice@test.com",
                skills={"python": 0.9, "sql": 0.7},
            ),
            StudentProfile(
                student_id="s2",
                name="Bob",
                email="bob@test.com",
                skills={"react": 0.8, "typescript": 0.6},
            ),
        ]
        state = _make_state(project_brief=SAMPLE_BRIEF, student_profiles=profiles)
        task_decomposition(state)

        # Inspect the messages passed to llm.invoke
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg_content: str = call_args[1]["content"]
        assert "python" in user_msg_content
        assert "react" in user_msg_content
        assert "sql" in user_msg_content
        assert "typescript" in user_msg_content

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_no_student_profiles_still_works(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF, student_profiles=[])
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5


# =========================================================================
# Task Decomposition — retry logic
# =========================================================================


class TestTaskDecompositionRetry:
    """Retry behaviour on malformed LLM output."""

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_retry_succeeds_after_bad_json(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm(["not valid json {{{", VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5
        assert mock_llm.invoke.call_count == 2

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_all_retries_exhausted(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm(["bad1", "bad2", "bad3"])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert result["task_array"] == []
        assert result["dependency_graph"] == {}
        assert mock_llm.invoke.call_count == 3

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_retry_on_missing_tasks_key(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        # Valid JSON but wrong schema (no "tasks" key)
        bad_schema = json.dumps({"items": [{"id": "task-1", "title": "Test"}]})
        mock_llm = _make_mock_llm([bad_schema, VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5
        assert mock_llm.invoke.call_count == 2


# =========================================================================
# Task Decomposition — parsing edge cases
# =========================================================================


class TestTaskDecompositionParsing:
    """JSON extraction edge cases."""

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_markdown_fences_stripped(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        fenced = f"```json\n{VALID_TASKS_JSON}\n```"
        mock_llm = _make_mock_llm([fenced])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_preamble_text_before_json(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        preamble = f"Here is the decomposition:\n{VALID_TASKS_JSON}"
        mock_llm = _make_mock_llm([preamble])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_duplicate_task_ids_trigger_retry(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        dup_json = json.dumps(
            {
                "tasks": [
                    {"id": "task-a", "title": "Task A", "effort_hours": 2.0},
                    {"id": "task-a", "title": "Task A copy", "effort_hours": 3.0},
                ]
            }
        )
        mock_llm = _make_mock_llm([dup_json, VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        assert len(result["task_array"]) == 5
        assert mock_llm.invoke.call_count == 2


# =========================================================================
# Task Decomposition — dependency graph
# =========================================================================


class TestTaskDecompositionDependencyGraph:
    """Dependency graph construction and validation."""

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_dependency_graph_keys_match_task_ids(
        self, mock_factory: MagicMock
    ) -> None:
        from agents.task_decomposition import task_decomposition

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        task_ids = {t.id for t in result["task_array"]}
        graph_keys = set(result["dependency_graph"].keys())
        assert task_ids == graph_keys

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_dangling_dependency_dropped(self, mock_factory: MagicMock) -> None:
        from agents.task_decomposition import task_decomposition

        dangling_json = json.dumps(
            {
                "tasks": [
                    {
                        "id": "task-a",
                        "title": "Task A",
                        "effort_hours": 2.0,
                        "dependencies": ["task-nonexistent"],
                    },
                    {
                        "id": "task-b",
                        "title": "Task B",
                        "effort_hours": 3.0,
                        "dependencies": ["task-a"],
                    },
                ]
            }
        )
        mock_llm = _make_mock_llm([dangling_json])
        mock_factory.return_value = mock_llm

        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = task_decomposition(state)

        dep_graph: dict[str, list[str]] = result["dependency_graph"]
        # "task-nonexistent" should have been dropped
        assert dep_graph["task-a"] == []
        assert dep_graph["task-b"] == ["task-a"]


# =========================================================================
# Graph integration — node replacement
# =========================================================================


class TestGraphNodeReplacement:
    """Verify the real agent is wired into the compiled graph."""

    def test_task_decomposition_node_is_real_implementation(self) -> None:
        """The graph's task_decomposition node should be the real agent,
        not the old placeholder that always returned ``{}``."""
        from graph.main import _NODE_FUNCTIONS

        node_fn = _NODE_FUNCTIONS["task_decomposition"]
        # The real implementation is imported from agents.task_decomposition
        assert node_fn.__module__ == "agents.task_decomposition"
        assert node_fn.__name__ == "task_decomposition"

    @patch("agents.task_decomposition.get_high_tier_llm")
    def test_real_node_returns_tasks_on_valid_state(
        self, mock_factory: MagicMock
    ) -> None:
        """Calling the graph's task_decomposition node with a valid brief
        should produce tasks (not an empty dict like the placeholder)."""
        from graph.main import _NODE_FUNCTIONS

        mock_llm = _make_mock_llm([VALID_TASKS_JSON])
        mock_factory.return_value = mock_llm

        node_fn = _NODE_FUNCTIONS["task_decomposition"]
        state = _make_state(project_brief=SAMPLE_BRIEF)
        result = node_fn(state)

        assert len(result["task_array"]) == 5
        assert "dependency_graph" in result
