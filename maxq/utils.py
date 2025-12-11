def normalize_dataset_name(name: str) -> str:
    """
    Normalizes the dataset name.
    """
    if not name:
        return ""
    return name.strip()
