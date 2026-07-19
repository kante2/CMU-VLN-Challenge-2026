#!/usr/bin/env python3
"""Base handler interface shared by the three question-type handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the node (and rclpy) at type-check time
    from vln_orchestrator.orchestrator_node import VLNOrchestrator


class BaseHandler:
    """A handler answers one question type by driving perception/reasoning and
    publishing through the node's publish_* helpers.

    Contract:
      - handle(question): run the real pipeline and publish the answer.
      - fallback(question): always publish *some* valid response (never leave the
        evaluator with nothing — partial credit beats silence, and an empty
        answer scores 0 with no chance of the time bonus).
    """

    def __init__(self, node: "VLNOrchestrator") -> None:
        self.node = node
        self.log = node.get_logger()

    def handle(self, question: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def fallback(self, question: str) -> None:  # pragma: no cover - interface
        raise NotImplementedError
