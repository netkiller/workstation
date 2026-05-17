import os


def current_edition():
    return os.environ.get("YOLOUTILS_EDITION", "community").strip().lower() or "community"


def is_community_edition():
    return current_edition() == "community"
