"""Unit tests for readaptation."""


import unittest

from transformers import AutoModelForCausalLM

from transformers_readapt import readapt


class TestReadapt(unittest.TestCase):
    def setUp(self):
        self.base_model = AutoModelForCausalLM.from_pretrained("afmck/testing-llama-tiny")
        self.finetuned_model = AutoModelForCausalLM.from_pretrained("afmck/testing-llama-tiny")
        self.instruction_model = AutoModelForCausalLM.from_pretrained("afmck/testing-llama-tiny")

    def test_readapt(self):
        combined_model = readapt(self.base_model, self.finetuned_model, self.instruction_model)
