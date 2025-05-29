import torch

def first_order_update(model, optimizer, X_current, Y_current, grad_clip, gradient_accumulation_steps, ctx_obj, get_batch_fn):
    """
    Performs a standard first-order optimization update (Adam or SGD).
    Handles gradient accumulation, gradient clipping, and fetching new batches.

    Args:
        model: The model to train.
        optimizer: The optimizer (e.g., AdamW).
        X_current, Y_current: The current batch of data for the first micro-step.
        grad_clip: Value for gradient clipping (0.0 to disable).
        gradient_accumulation_steps: Number of micro-steps to accumulate gradients over.
        ctx_obj: AMP context (e.g., torch.amp.autocast or nullcontext).
        get_batch_fn: A function that takes a split (e.g., 'train') and returns (X, Y) a new batch.

    Returns:
        loss: The scaled loss from the last micro-step.
        X_next, Y_next: The batch fetched for the *next* iteration's first micro-step.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True) # Zero out gradients at the start of the accumulation
    accumulated_loss = 0.0

    # Use the provided X_current, Y_current for the first micro-step
    X_micro, Y_micro = X_current, Y_current

    for micro_step in range(gradient_accumulation_steps):
        with ctx_obj:
            logits, loss_micro = model(X_micro, Y_micro)
        
        # Scale the loss for accumulation
        loss_micro_scaled = loss_micro / gradient_accumulation_steps
        accumulated_loss += loss_micro_scaled.item() # Accumulate for returning average loss
        
        # Backward pass on the scaled loss
        loss_micro_scaled.backward()
        
        # Fetch the next batch for the next micro-step, *except* for the last micro-step
        # For the last micro-step, this batch will be used for the *next* main iteration
        if micro_step < gradient_accumulation_steps - 1:
            X_micro, Y_micro = get_batch_fn('train')
        else:
            # This is the batch for the next iteration
            X_next, Y_next = get_batch_fn('train')
    
    # Clip gradients after accumulation
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    # Step the optimizer
    optimizer.step()
    
    # Zero out gradients again, ready for the next accumulation cycle
    optimizer.zero_grad(set_to_none=True)
    
    # Return the loss from the last micro_batch (already scaled) and the prefetched X,Y for the next iter
    return loss_micro_scaled, X_next, Y_next 