from abc import ABC, abstractmethod
from pathlib import Path


class BaseMLClassifier(ABC):
    name: str

    @abstractmethod
    def train(self, texts: list[str], axes: list[dict], labels: dict[str, list]) -> None:
        """
        texts: list of training texts
        axes: axis defs from meta_data.jsonl
        labels: {axis_en: [label_per_story, ...]}
                For multi-label axes, each element is a list[str].
        """

    @abstractmethod
    def predict_one(self, text: str, axes: list[dict]) -> dict[str, object]:
        """Returns {axis_en: predicted_value_or_list} for a single text."""

    @abstractmethod
    def save(self, model_dir: Path) -> None:
        """Serialize all artifacts to model_dir/model.pkl."""

    @classmethod
    @abstractmethod
    def load(cls, model_dir: Path) -> "BaseMLClassifier":
        """Deserialize from model_dir/model.pkl."""
