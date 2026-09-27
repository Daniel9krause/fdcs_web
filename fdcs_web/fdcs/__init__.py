"""FDCS core package: model, detector, knowledge base and storage."""
import json


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
