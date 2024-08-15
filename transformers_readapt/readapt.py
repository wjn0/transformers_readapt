from copy import deepcopy

from torch import nn

from transformers.models.llama.modeling_llama import LlamaForCausalLM


def readapt(finetuned_model, instruction_model, weight=0.5):
    """
    Combine a finetuned model with an instruction-tuned model.

    Args:
        finetuned_model: A finetuned model.
        instruction_model: An instruction-tuned model.
        weight: The weight of the finetuned model.

    Returns:
        combined_model: The combined model.
    """
    assert type(finetuned_model) == type(instruction_model), "Models must be of the same type."

    if not isinstance(finetuned_model, nn.Module) or not isinstance(instruction_model, nn.Module):
        raise ValueError("Models must be PyTorch models.")

    if not isinstance(finetuned_model, LlamaForCausalLM):
        warn("Models are not Llama models. This function may not work as expected.")

    # Combine the weights of the models.
    combined_model = deepcopy(finetuned_model)
    for param, instruction_param in zip(combined_model.parameters(), instruction_model.parameters()):
        param.data = weight * param.data + (1 - weight) * instruction_param.data

    return combined_model
