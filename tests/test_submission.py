"""The submission's own claims, enforced rather than asserted in prose.

``docs/submission_rules.md`` lists five required artifacts and a set of
reproducibility requirements. Three of the five did not exist for three
phases (D Phase 15 review, item 1), and the strongest property this
submission has -- stdlib-only, offline, deterministic -- was true the whole
time and written down nowhere.

Prose cannot hold a claim like that: someone adds ``import requests`` in a
later phase and every document saying "no network" becomes false silently.
So the claims are tests. If the submission stops being offline, this file
goes red before the disclosure goes stale.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import sysconfig
import unittest
from pathlib import Path

import starter

_ROOT = Path(__file__).resolve().parent.parent
_PACKAGE = Path(starter.__file__).resolve().parent

# Modules that would mean a network call, a credential, or a model. None of
# these may appear anywhere in the shipped package.
_FORBIDDEN = {
    "requests", "urllib", "urllib3", "http", "httpx", "socket", "ssl",
    "ftplib", "smtplib", "telnetlib", "xmlrpc", "asyncio", "aiohttp",
    "openai", "anthropic", "google", "boto3", "torch", "transformers",
    "sentence_transformers", "numpy", "scipy", "sklearn", "pandas",
    "subprocess", "os",
}


def _is_stdlib(name: str) -> bool:
    """True if ``name`` resolves to the standard library of THIS interpreter.

    Not ``sys.stdlib_module_names``: that is 3.10+, and this submission
    certifies on 3.9.6, so using it made the suite ERROR on the exact
    interpreter SUBMISSION.md claims it passes on (D Phase 15 review, item 1).
    A test that cannot run where the claim is made is worse than no test.

    Resolving the spec is also a stronger check than a name lookup. A
    third-party package that shadows a stdlib name would pass a name test and
    fails this one, because the resolved origin lands in site-packages
    instead of the stdlib directory.
    """
    if name in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return False
    if spec is None:
        return False
    if spec.origin in ("built-in", "frozen"):
        return True
    if spec.origin is None:                       # namespace package
        return False
    origin = Path(spec.origin).resolve()
    if "site-packages" in origin.parts or "dist-packages" in origin.parts:
        return False
    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    try:
        origin.relative_to(stdlib)
    except ValueError:
        return False
    return True


def _imports(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module.split(".")[0])
    return found


class ShippedPackageIsStdlibOnlyTest(unittest.TestCase):
    """The claim the whole submission rests on."""

    def _package_imports(self) -> set[str]:
        found: set[str] = set()
        for path in sorted(_PACKAGE.glob("*.py")):
            found |= _imports(path)
        return found

    def test_no_third_party_dependency(self) -> None:
        external = {
            name for name in self._package_imports()
            if name not in {"starter", "__future__"} and not _is_stdlib(name)
        }
        self.assertEqual(external, set(),
                         f"starter/ now depends on {sorted(external)}; "
                         "requirements.txt and the disclosure both say it "
                         "depends on nothing")

    def test_nothing_that_could_reach_the_network(self) -> None:
        reachable = self._package_imports() & _FORBIDDEN
        self.assertEqual(reachable, set(),
                         f"starter/ imports {sorted(reachable)}; the "
                         "submission documents itself as offline with no "
                         "credentials")

    def test_no_model_call_in_the_turn_path(self) -> None:
        # The token disclosure reports a literal {0, 0}. That is only honest
        # while nothing in the package can spend a token.
        joined = "\n".join(
            path.read_text(encoding="utf-8") for path in _PACKAGE.glob("*.py"))
        for marker in ("api_key", "API_KEY", "openai", "anthropic",
                       "chat.completions", "http://", "https://api."):
            self.assertNotIn(marker, joined,
                             f"{marker!r} appears in the shipped package")


class RequiredArtifactsExistTest(unittest.TestCase):
    """The five things docs/submission_rules.md asks every team to submit."""

    def test_the_agent_entry_file_exports_agent(self) -> None:
        from starter.agent import Agent

        self.assertTrue(hasattr(Agent, "reset"))
        self.assertTrue(hasattr(Agent, "respond"))

    def test_setup_instructions_exist(self) -> None:
        self.assertTrue((_ROOT / "SUBMISSION.md").is_file())

    def test_the_method_and_limitations_report_exists(self) -> None:
        self.assertTrue((_ROOT / "docs" / "method_and_limitations.md").is_file())

    def test_the_performance_disclosure_exists(self) -> None:
        self.assertTrue((_ROOT / "docs" / "performance_disclosure.md").is_file())

    def test_a_dependency_manifest_exists(self) -> None:
        # Required even though it is empty: "no dependencies" and "nobody
        # wrote the file" are indistinguishable to an organizer otherwise.
        self.assertTrue((_ROOT / "requirements.txt").is_file())

    def test_the_setup_instructions_state_an_exact_python_version(self) -> None:
        # docs/submission_rules.md: "exact Python version requirement if
        # non-default". Every measurement in this repo is certified on 3.9.6
        # while the organizer's README only says "3.10 or later is
        # recommended", so the version IS non-default and must be stated.
        text = (_ROOT / "SUBMISSION.md").read_text(encoding="utf-8")
        self.assertIn("3.9.6", text)
        self.assertIn("python3 -m evaluator.local_evaluator", text)


class ResponseContractTest(unittest.TestCase):
    """The output rules, checked against the organizer's own schema."""

    def test_ask_attribute_enum_is_the_organizers(self) -> None:
        contract = json.loads(
            (_ROOT / "docs" / "agent_api_contract.json").read_text(
                encoding="utf-8"))
        enum = contract["turn_response"]["properties"]["ask_attribute"]["enum"]
        from starter.clarify import ALLOWED_ATTRIBUTES

        self.assertEqual(set(ALLOWED_ATTRIBUTES),
                         {value for value in enum if value is not None})

    def test_usage_is_reported_and_non_negative(self) -> None:
        from starter.agent import _to_response
        from starter.contracts import Candidate, RankingResult

        payload = _to_response(RankingResult(ranked=[Candidate("A")]), 10)
        usage = payload["usage"]
        self.assertGreaterEqual(usage["prompt_tokens"], 0)
        self.assertGreaterEqual(usage["completion_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
