from typing import Any


def generate_dataset(*args: Any, **kwargs: Any) -> Any:
    from simulator.data_generator.generate import generate_dataset as _generate_dataset

    return _generate_dataset(*args, **kwargs)


__all__ = ["generate_dataset"]
