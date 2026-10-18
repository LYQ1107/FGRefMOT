"""Small compatibility shims for the pinned upstream loaders.

FlexHook's pinned loader still calls ``batch_encode_plus``.  Transformers 5
removed that public alias while keeping the equivalent ``__call__`` API.  The
shim is installed only by our research launch/smoke entry points; the vendored
upstream checkout remains untouched.
"""

from __future__ import annotations

from typing import Any


def install_legacy_tokenizer_alias() -> bool:
    """Restore the removed tokenizer method when the installed API needs it.

    Returns ``True`` when this call adds the alias and ``False`` when the
    installed Transformers version already provides it.
    """

    from transformers import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        return False

    def batch_encode_plus(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The legacy method truncated when a finite ``max_length`` was passed
        # with ``padding="max_length"``.  Transformers 5 requires the
        # truncation flag explicitly, so preserve the pinned loader's intent.
        if kwargs.get("max_length") is not None and "truncation" not in kwargs:
            kwargs["truncation"] = True
        return self(*args, **kwargs)

    setattr(PreTrainedTokenizerBase, "batch_encode_plus", batch_encode_plus)
    return True
