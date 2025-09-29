import json
from typing import Any

def serialize_for_json(obj: Any) -> Any:
    """Convert objects to JSON-serializable format."""
    if hasattr(obj, '__dict__'):
        return {
            '_type': obj.__class__.__name__,
            '_module': obj.__class__.__module__,
            'data': {k: serialize_for_json(v) for k, v in obj.__dict__.items()}
        }
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return {'_type': 'set', 'data': [serialize_for_json(item) for item in obj]}
    else:
        return obj

def deserialize_from_json(obj: Any) -> Any:
    """Reconstruct objects from JSON format."""
    if isinstance(obj, dict) and '_type' in obj:
        if obj['_type'] == 'set':
            return set(deserialize_from_json(item) for item in obj['data'])
        else:
            # For other custom objects, return as dict
            return {k: deserialize_from_json(v) for k, v in obj.get('data', obj).items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: deserialize_from_json(v) for k, v in obj.items()}
    else:
        return obj
