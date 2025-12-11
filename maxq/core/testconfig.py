"""
MaxQ Test Configuration - YAML-based declarative test definitions.

Example maxq.yaml:
```yaml
description: "E-commerce product search"

provider:
  collection: products
  model: mixedbread-ai/mxbai-embed-large-v1

defaults:
  top_k: 10

tests:
  - query: "waterproof hiking boots"
    assert:
      - type: contains-id
        value: "product-123"
      - type: ndcg
        threshold: 0.8
      - type: latency
        max_ms: 200

  - query: "{{color}} {{item}}"
    vars:
      - { color: "red", item: "dress" }
      - { color: "blue", item: "jeans" }
    ground_truth: ["doc-1", "doc-2"]
    assert:
      - type: recall
        threshold: 0.9
```
"""

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class AssertionConfig(BaseModel):
    """Configuration for a single assertion."""

    type: str
    # Common parameters
    threshold: float | None = None
    value: str | list[str] | None = None
    k: int | None = None
    max_ms: float | None = None
    min_ms: float | None = None
    # LLM assertion parameters
    model: str | None = None
    rubric: str | None = None
    # Semantic assertion parameters
    expected: str | None = None
    min_diversity: float | None = None
    # Field-based assertion parameters
    field: str | None = None
    pattern: str | None = None
    min_value: float | None = None
    max_value: float | None = None


class TestCase(BaseModel):
    """A single test case definition."""

    query: str
    description: str | None = None
    filters: dict[str, Any] | None = None
    ground_truth: list[str] | None = None
    ground_truth_file: str | None = None
    top_k: int | None = None
    tags: list[str] = Field(default_factory=list)
    vars: list[dict[str, str]] | None = None  # For variable expansion
    assertions: list[AssertionConfig] = Field(default_factory=list, alias="assert")

    class Config:
        populate_by_name = True


class ProviderConfig(BaseModel):
    """Qdrant provider configuration."""

    collection: str
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str | None = None
    fusion: str | None = None  # "rrf" for hybrid search
    url: str | None = None  # Override env QDRANT_URL
    api_key: str | None = None  # Override env QDRANT_API_KEY


class DefaultsConfig(BaseModel):
    """Default settings for all tests."""

    top_k: int = 10
    timeout_ms: float = 30000


class MaxQConfig(BaseModel):
    """Root configuration for maxq.yaml."""

    description: str | None = None
    provider: ProviderConfig
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    tests: list[TestCase] = Field(default_factory=list)

    @field_validator("tests", mode="before")
    @classmethod
    def expand_test_variables(cls, tests: list[dict]) -> list[dict]:
        """Expand tests with variable substitution."""
        expanded = []
        for test in tests:
            if "vars" in test and test["vars"]:
                # Expand each variable combination into a separate test
                for var_set in test["vars"]:
                    new_test = test.copy()
                    new_test.pop("vars", None)
                    # Substitute variables in query
                    query = new_test["query"]
                    for key, val in var_set.items():
                        query = query.replace(f"{{{{{key}}}}}", str(val))
                    new_test["query"] = query
                    # Also substitute in description if present
                    if "description" in new_test and new_test["description"]:
                        desc = new_test["description"]
                        for key, val in var_set.items():
                            desc = desc.replace(f"{{{{{key}}}}}", str(val))
                        new_test["description"] = desc
                    expanded.append(new_test)
            else:
                expanded.append(test)
        return expanded


def substitute_env_vars(content: str) -> str:
    """Replace ${VAR} and $VAR patterns with environment variable values."""
    # Match ${VAR} or ${VAR:-default}
    def replace_match(match: re.Match) -> str:
        var_expr = match.group(1)
        if ":-" in var_expr:
            var_name, default = var_expr.split(":-", 1)
            return os.environ.get(var_name, default)
        return os.environ.get(var_expr, match.group(0))

    # Replace ${VAR} and ${VAR:-default} patterns
    content = re.sub(r"\$\{([^}]+)\}", replace_match, content)
    return content


def load_config(path: str | Path = "maxq.yaml") -> MaxQConfig:
    """
    Load and parse a maxq.yaml configuration file.

    Args:
        path: Path to the config file (default: maxq.yaml in current dir)

    Returns:
        Parsed MaxQConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        content = f.read()

    # Substitute environment variables
    content = substitute_env_vars(content)

    # Parse YAML
    data = yaml.safe_load(content)

    if not data:
        raise ValueError(f"Empty or invalid config file: {path}")

    return MaxQConfig(**data)


def validate_config(path: str | Path = "maxq.yaml") -> tuple[bool, list[str]]:
    """
    Validate a maxq.yaml configuration file.

    Args:
        path: Path to the config file

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []

    try:
        config = load_config(path)

        # Check provider configuration
        if not config.provider.collection:
            errors.append("Provider collection name is required")

        # Check tests
        if not config.tests:
            errors.append("No tests defined")

        for i, test in enumerate(config.tests):
            if not test.query:
                errors.append(f"Test {i + 1}: query is required")

            for j, assertion in enumerate(test.assertions):
                # Validate assertion types
                valid_types = {
                    "contains-id",
                    "not-empty",
                    "count",
                    "ndcg",
                    "mrr",
                    "recall",
                    "precision",
                    "hit-rate",
                    "latency",
                    "contains-text",
                    "regex",
                    "field-equals",
                    "field-range",
                    "llm-relevance",
                    "llm-rubric",
                    "semantic-similarity",
                    "semantic-diversity",
                }
                if assertion.type not in valid_types:
                    errors.append(
                        f"Test {i + 1}, assertion {j + 1}: "
                        f"unknown type '{assertion.type}'"
                    )

                # Validate required parameters for specific types
                if assertion.type == "contains-id" and not assertion.value:
                    errors.append(
                        f"Test {i + 1}, assertion {j + 1}: "
                        "contains-id requires 'value' parameter"
                    )

                if assertion.type in {"ndcg", "mrr", "recall", "precision", "hit-rate"}:
                    if assertion.threshold is None:
                        errors.append(
                            f"Test {i + 1}, assertion {j + 1}: "
                            f"{assertion.type} requires 'threshold' parameter"
                        )
                    # These also need ground_truth
                    if not test.ground_truth and not test.ground_truth_file:
                        errors.append(
                            f"Test {i + 1}, assertion {j + 1}: "
                            f"{assertion.type} requires ground_truth"
                        )

                if assertion.type == "latency" and assertion.max_ms is None:
                    errors.append(
                        f"Test {i + 1}, assertion {j + 1}: "
                        "latency requires 'max_ms' parameter"
                    )

    except FileNotFoundError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Configuration error: {e}")

    return len(errors) == 0, errors


def generate_example_config() -> str:
    """Generate an example maxq.yaml configuration."""
    return '''# maxq.yaml - Retrieval Evaluation Configuration
description: "Example retrieval test suite"

# Qdrant provider configuration
provider:
  collection: my_collection
  model: sentence-transformers/all-MiniLM-L6-v2
  # sparse_model: Qdrant/bm25  # Uncomment for hybrid search
  # fusion: rrf

# Default settings for all tests
defaults:
  top_k: 10

# Test cases
tests:
  # Basic test - check that specific docs are returned
  - query: "example search query"
    description: "Test basic search functionality"
    tags: [smoke, basic]
    ground_truth:
      - "doc-id-1"
      - "doc-id-2"
    assert:
      - type: not-empty
      - type: contains-id
        value: "doc-id-1"
      - type: recall
        threshold: 0.8
      - type: latency
        max_ms: 500

  # Test with variable expansion
  - query: "{{category}} products under $100"
    vars:
      - { category: "electronics" }
      - { category: "clothing" }
      - { category: "home" }
    assert:
      - type: not-empty
      - type: latency
        max_ms: 1000

  # LLM-judged relevance (requires OPENAI_API_KEY)
  # - query: "find similar products"
  #   assert:
  #     - type: llm-relevance
  #       model: gpt-4o-mini
  #       threshold: 0.7
'''
