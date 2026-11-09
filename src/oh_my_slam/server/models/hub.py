"""Model files from the Hugging Face cache, without a request to the Hub when they are there."""

from __future__ import annotations


def cached_file(repo_id: str, filename: str, revision: str | None = None) -> str | None:
    """The cached path of ``filename`` of ``repo_id`` at ``revision`` (default: the cached
    ``main``), read offline; None when it is not cached (the caller then downloads it)."""
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id, filename, revision=revision, local_files_only=True)
    except Exception:  # LocalEntryNotFoundError, or a cache that cannot be read
        return None
