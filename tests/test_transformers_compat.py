import unittest

from transformers import PreTrainedTokenizerBase

from scripts.transformers_compat import install_legacy_tokenizer_alias


class TransformersCompatTest(unittest.TestCase):
    def test_legacy_tokenizer_alias_is_available(self):
        install_legacy_tokenizer_alias()
        self.assertTrue(hasattr(PreTrainedTokenizerBase, "batch_encode_plus"))


if __name__ == "__main__":
    unittest.main()
