from pathlib import Path

from scripts.validate_results import validate_repository


def test_committed_showcase_evidence_matches_readme() -> None:
    validate_repository(Path(__file__).resolve().parents[1])
