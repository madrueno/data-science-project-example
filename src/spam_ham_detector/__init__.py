"""Spam/Ham detector package for YouTube comment classification."""

__version__ = "0.1.0"

from spam_ham_detector.dataset import CommentsDataset
from spam_ham_detector.evaluation import evaluate_model

__all__ = ["CommentsDataset", "evaluate_model"]
