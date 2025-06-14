import torch

def build_causal_attention_mask(
        num_text_tokens: int,
        num_frames: int,
        tokens_per_frame: int,
        current_frame_idx: int,
        causal_window: int,
        device: torch.device = "cpu"
):
    """
    Constructs a causal attention mask based on your requirements.
    """

    total_tokens = num_text_tokens + num_frames * tokens_per_frame
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)

    # Step 1: Text tokens attend only to themselves
    text_range = slice(0, num_text_tokens)
    mask[text_range, text_range] = True

    # Step 2: Visual tokens attend to:
    #   - all text tokens
    #   - first frame visual tokens
    #   - up to N past frame visual tokens
    #   - own frame visual tokens

    for t in range(num_frames):
        # Compute global index range for current frame
        start_t = num_text_tokens + t * tokens_per_frame
        end_t = start_t + tokens_per_frame
        visual_slice = slice(start_t, end_t)

        allowed_indices = []

        # Always allow attention to all text tokens
        allowed_indices += list(range(0, num_text_tokens))

        # Always allow first frame visual tokens
        first_frame_start = num_text_tokens
        first_frame_end = first_frame_start + tokens_per_frame
        allowed_indices += list(range(first_frame_start, first_frame_end))

        # Allow attention to previous N frames and current frame
        for past_t in range(max(1, t - causal_window + 1), t + 1):  # exclude frame 0 here
            past_start = num_text_tokens + past_t * tokens_per_frame
            past_end = past_start + tokens_per_frame
            allowed_indices += list(range(past_start, past_end))

        # Update mask
        for row in range(start_t, end_t):
            mask[row, allowed_indices] = True

    return mask  # shape: (total_seq_len, total_seq_len)
