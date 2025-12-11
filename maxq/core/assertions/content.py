"""Content-based assertions: contains-text, regex, field-equals, field-range."""

import re

from maxq.core.assertions.base import Assertion, AssertionResult, EvalContext, SearchResult
from maxq.core.assertions.registry import register_assertion


@register_assertion("contains-text")
class ContainsTextAssertion(Assertion):
    """Assert that results contain specific text."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        expected_text = self.config.get("value", "")
        case_sensitive = self.config.get("case_sensitive", False)

        if not case_sensitive:
            expected_text = expected_text.lower()

        matches = []
        for r in results:
            text = r.text or ""
            if not case_sensitive:
                text = text.lower()
            if expected_text in text:
                matches.append(r.id)

        passed = len(matches) > 0
        return self._make_result(
            passed=passed,
            message=f"Found '{expected_text}' in {len(matches)} results"
            if passed
            else f"Text '{expected_text}' not found in any result",
            expected=f"Contains: '{expected_text}'",
            actual=f"{len(matches)} matches",
            details={"matching_ids": matches},
        )


@register_assertion("regex")
class RegexAssertion(Assertion):
    """Assert that results match a regex pattern."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        pattern = self.config.get("pattern", "")
        flags = 0
        if not self.config.get("case_sensitive", False):
            flags = re.IGNORECASE

        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return self._make_result(
                passed=False,
                message=f"Invalid regex pattern: {e}",
                expected=pattern,
                actual="Regex error",
            )

        matches = []
        for r in results:
            text = r.text or ""
            if compiled.search(text):
                matches.append(r.id)

        passed = len(matches) > 0
        return self._make_result(
            passed=passed,
            message=f"Pattern matched in {len(matches)} results"
            if passed
            else f"Pattern '{pattern}' not found",
            expected=f"Matches: /{pattern}/",
            actual=f"{len(matches)} matches",
            details={"matching_ids": matches},
        )


@register_assertion("field-equals")
class FieldEqualsAssertion(Assertion):
    """Assert that a metadata field equals a specific value."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        field = self.config.get("field", "")
        expected_value = self.config.get("value")
        all_match = self.config.get("all", False)  # Require all results to match

        if not field:
            return self._make_result(
                passed=False,
                message="field-equals requires 'field' parameter",
                expected="Field name",
                actual="None",
            )

        matches = []
        mismatches = []
        for r in results:
            metadata = r.metadata or {}
            actual = metadata.get(field)
            if actual == expected_value:
                matches.append(r.id)
            else:
                mismatches.append({"id": r.id, "value": actual})

        if all_match:
            passed = len(mismatches) == 0 and len(matches) > 0
        else:
            passed = len(matches) > 0

        return self._make_result(
            passed=passed,
            message=f"{len(matches)}/{len(results)} results have {field}={expected_value}",
            expected=f"{field} = {expected_value}",
            actual=f"{len(matches)} matches",
            details={"matching_ids": matches, "mismatches": mismatches[:5]},
        )


@register_assertion("field-range")
class FieldRangeAssertion(Assertion):
    """Assert that a numeric metadata field is within a range."""

    def evaluate(
        self,
        results: list[SearchResult],
        context: EvalContext,
    ) -> AssertionResult:
        field = self.config.get("field", "")
        min_val = self.config.get("min_value")
        max_val = self.config.get("max_value")
        all_match = self.config.get("all", False)

        if not field:
            return self._make_result(
                passed=False,
                message="field-range requires 'field' parameter",
                expected="Field name",
                actual="None",
            )

        if min_val is None and max_val is None:
            return self._make_result(
                passed=False,
                message="field-range requires 'min_value' or 'max_value'",
                expected="Range bounds",
                actual="None",
            )

        matches = []
        out_of_range = []
        for r in results:
            metadata = r.metadata or {}
            value = metadata.get(field)

            try:
                value = float(value) if value is not None else None
            except (ValueError, TypeError):
                value = None

            if value is None:
                out_of_range.append({"id": r.id, "value": None})
                continue

            in_range = True
            if min_val is not None and value < min_val:
                in_range = False
            if max_val is not None and value > max_val:
                in_range = False

            if in_range:
                matches.append(r.id)
            else:
                out_of_range.append({"id": r.id, "value": value})

        if all_match:
            passed = len(out_of_range) == 0 and len(matches) > 0
        else:
            passed = len(matches) > 0

        range_str = ""
        if min_val is not None and max_val is not None:
            range_str = f"{min_val} <= {field} <= {max_val}"
        elif min_val is not None:
            range_str = f"{field} >= {min_val}"
        else:
            range_str = f"{field} <= {max_val}"

        return self._make_result(
            passed=passed,
            message=f"{len(matches)}/{len(results)} results satisfy {range_str}",
            expected=range_str,
            actual=f"{len(matches)} in range",
            details={"matching_ids": matches, "out_of_range": out_of_range[:5]},
        )
