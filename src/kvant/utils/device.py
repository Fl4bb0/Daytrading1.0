from __future__ import annotations


def resolve_torch_device(requested: str | None = "auto") -> str:
    """Resolve a configured torch device string.

    ``auto`` chooses the fastest commonly available backend in this order:
    CUDA, Apple MPS, then CPU. Explicit devices are validated so configuration
    mistakes fail early with a useful error.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project depends on torch
        raise RuntimeError("PyTorch is required to resolve train.device") from exc

    value = str(requested or "auto").strip().lower()
    if value in {"", "auto", "best"}:
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    if value.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Configured train.device={requested!r}, but CUDA is not available")
        if ":" in value:
            try:
                index = int(value.split(":", 1)[1])
            except ValueError as exc:
                raise RuntimeError(f"Invalid CUDA device string: {requested!r}") from exc
            if index < 0 or index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Configured train.device={requested!r}, but only "
                    f"{torch.cuda.device_count()} CUDA device(s) are available"
                )
        return value

    if value == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError(f"Configured train.device={requested!r}, but Apple MPS is not available")
        return value

    if value == "cpu":
        return value

    try:
        torch.device(value)
    except Exception as exc:
        raise RuntimeError(f"Invalid train.device value: {requested!r}") from exc
    return value
