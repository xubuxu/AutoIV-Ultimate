from dataclasses import dataclass, field
import json
import os

@dataclass
class Config:
    """Unified configuration for AutoIV-Ultimate.
    This merges settings from the original projects.
    """
    # Example fields – can be extended later
    data_folder: str = ""
    output_folder: str = "output"
    theme: str = "dark"
    # Add more configuration options as needed

    @classmethod
    def load(cls, path: str = "config.json"):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: str = "config.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2)
