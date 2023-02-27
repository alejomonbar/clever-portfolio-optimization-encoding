import numpy as np
import json


all_instances = None


def to_list(d: dict) -> dict:
    """convert numpy.ndarray objects in dict to lists for serialization"""
    return {
        key: to_list(value)
        if isinstance(value, dict)
        else (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in d.items()
    }


def to_array(d: dict) -> dict:
    """convert lists in dict to numpy.ndarray objects for deserialization"""
    return {
        int(key)
        if isinstance(key, str) and key.isdigit()
        else key: to_array(value)
        if isinstance(value, dict)
        else (np.array(value) if isinstance(value, list) else value)
        for key, value in d.items()
    }


def load_instances():
    """load problem instances from file"""
    with open("problem_instances.json", "r") as f:
        return to_array(json.load(f))


def get_instance(instance_type: str, num_assets: int) -> dict:
    """get one problem instance"""
    global all_instances
    if all_instances is None:
        all_instances = load_instances()
    return all_instances[instance_type][num_assets]
