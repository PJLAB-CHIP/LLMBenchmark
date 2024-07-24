import torch


def gen_random_input_token(
    batch_size: int, seq_len: int, vocab_size: int
) -> torch.Tensor:
    """
    Generate random input tokens.

    Returns
    -------
    torch.Tensor
        Random input tokens with shape (batch_size, seq_len).
    """
    ids = torch.randint(low=0, high=vocab_size - 1, size=(batch_size, seq_len))
    return ids
