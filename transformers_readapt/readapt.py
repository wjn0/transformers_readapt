from warnings import warn
from copy import deepcopy

from torch import nn

from transformers.models.llama.modeling_llama import LlamaForCausalLM


def readapt(base_model, finetuned_model, instruction_model, finetuned_weight=0.5, instruction_weight=0.5):
    """
    Combine a finetuned model with an instruction-tuned model.

    Reference: https://arxiv.org/abs/2405.15007

    Args:
        base_model: The base model.
        finetuned_model: A finetuned model.
        instruction_model: An instruction-tuned model.
        finetuned_weight: The weight of the finetuned model.
        instruction_weight: The weight of the instruction-tuned model.

    Returns:
        combined_model: The combined model.
    """
    assert type(finetuned_model) == type(instruction_model), "Models must be of the same type."

    if not isinstance(finetuned_model, nn.Module) or not isinstance(instruction_model, nn.Module):
        raise ValueError("Models must be PyTorch models.")

    if not isinstance(finetuned_model, LlamaForCausalLM):
        warn("Models are not Llama models. This function may not work as expected.")

    # Combine the weights of the models according to the formula:
    #
    #     x = base_model + finetune_weight * (finetuned_model - base_model) + instruction_weight * (instruction_model - base_model)
    #
    # by updating the (deepcopy'd) fine-tuned model in-place.
    combined_model = deepcopy(finetuned_model)
    for base_param, param, instruction_param in zip(base_model.parameters(), combined_model.parameters(), instruction_model.parameters()):
        param.data = base_param.data + finetuned_weight * (param.data - base_param.data) + instruction_weight * (base_param.data - instruction_param.data)

    return combined_model
