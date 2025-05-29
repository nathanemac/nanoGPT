import torch
# Note: ctx_obj should be an initialized torch.amp.autocast or nullcontext
# It's passed to zo_forward.

def zo_forward(model, x, y, ctx_obj):
    """
    Get (no gradient) loss from the model.
    """
    model.eval()
    with torch.inference_mode():
        with ctx_obj: # Use the passed context object
            logits, loss = model(x, y)
    return loss.detach()

class AdaptiveZoEps:
    """
    Adaptive zo_eps manager that adjusts perturbation size based on:
    1. Success/failure in finding good directions (DiMeZO-specific)
    2. Learning rate coupling (eps scales with lr decay)
    """
    
    def __init__(self, base_eps=1e-3, base_lr=6e-4, window_size=20, 
                 eps_increase_rate=0.05, eps_decrease_rate=0.1,
                 lr_coupling_strength=0.5, min_eps=1e-6, max_eps=1e-3,
                 adaptive_eps_success_high=0.7, # For potential future use
                 adaptive_eps_success_low=0.3):  # For potential future use
        """
        Initialize adaptive zo_eps manager.
        
        Args:
            base_eps: Base perturbation size
            base_lr: Base learning rate for coupling
            window_size: Number of steps to track for success rate
            eps_increase_rate: Rate to increase eps when successful
            eps_decrease_rate: Rate to decrease eps when failing
            lr_coupling_strength: How strongly eps follows lr (0=no coupling, 1=full coupling)
            min_eps, max_eps: Bounds on eps values
            adaptive_eps_success_high: Success rate threshold for increasing eps (currently unused in provided logic)
            adaptive_eps_success_low: Success rate threshold for decreasing eps (currently unused in provided logic)
        """
        self.base_eps = base_eps
        self.base_lr = base_lr
        self.window_size = window_size
        self.eps_increase_rate = eps_increase_rate
        self.eps_decrease_rate = eps_decrease_rate
        self.lr_coupling_strength = lr_coupling_strength
        self.min_eps = min_eps
        self.max_eps = max_eps
        self.adaptive_eps_success_high = adaptive_eps_success_high
        self.adaptive_eps_success_low = adaptive_eps_success_low
        
        # State tracking
        self.current_eps = base_eps
        self.success_history = []
        self.step_count = 0
        
    def record_step(self, loss, projected_grad, learning_rate, method_info=None):
        """
        Record information from the current optimization step.
        
        Args:
            loss: Current loss value (not used, kept for compatibility)
            projected_grad: Projected gradient magnitude (not used, kept for compatibility)
            learning_rate: Current learning rate for coupling
            method_info: Dictionary with method-specific info, expects {'success': bool}
        """
        self.step_count += 1
        
        # Extract success from method_info
        if method_info and 'success' in method_info:
            is_successful = method_info['success']
        else:
            # Fallback: assume no success if info not provided
            is_successful = False
        
        # Update history with sliding window
        self.success_history.append(is_successful)
        
        # Keep only recent history
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # Update eps immediately based on this single step
        self._update_eps_immediate(is_successful, learning_rate)
    
    def _update_eps_immediate(self, success, learning_rate):
        """Update eps immediately based on current step success and learning rate."""
        
        if success:
            # Success: increase exploration (but cap at max_eps)
            adaptive_factor = 1 + self.eps_increase_rate
        else:
            # Failure: decrease exploration (but cap at min_eps)
            adaptive_factor = 1 - self.eps_decrease_rate
        
        # Learning rate coupling: eps should scale with lr
        # Prevent division by zero if base_lr is somehow 0
        lr_factor = (learning_rate / self.base_lr) ** self.lr_coupling_strength if self.base_lr != 0 else 1.0
        
        # Apply both adjustments
        new_eps = self.current_eps * adaptive_factor * lr_factor
        
        # Apply bounds
        self.current_eps = max(self.min_eps, min(self.max_eps, new_eps))
        
    def get_eps(self):
        """Get the current adaptive eps value."""
        return self.current_eps
    
    def get_stats(self):
        """Get statistics for logging/debugging."""
        if len(self.success_history) == 0:
            return {
                'eps': self.current_eps,
                'success_rate': 0.0,
                'steps_tracked': 0
            }
        
        return {
            'eps': self.current_eps,
            'success_rate': sum(self.success_history) / len(self.success_history),
            'steps_tracked': len(self.success_history)
        }

    def get_state(self):
        """Get the current state of the adaptive eps manager for checkpointing."""
        return {
            'current_eps': self.current_eps,
            'success_history': list(self.success_history), # Ensure it's a list for serialization
            'step_count': self.step_count,
            # Optional: save base parameters if they might change and you want to restore them
            # 'base_eps': self.base_eps,
            # 'base_lr': self.base_lr,
            # 'lr_coupling_strength': self.lr_coupling_strength,
            # 'min_eps': self.min_eps,
            # 'max_eps': self.max_eps,
            # 'window_size': self.window_size,
            # 'adaptive_eps_success_high': self.adaptive_eps_success_high,
            # 'adaptive_eps_success_low': self.adaptive_eps_success_low,
            # 'eps_increase_rate': self.eps_increase_rate,
            # 'eps_decrease_rate': self.eps_decrease_rate
        }

    def load_state(self, state):
        """Load the state of the adaptive eps manager from a checkpoint."""
        self.current_eps = state.get('current_eps', self.current_eps)
        # Ensure history is a list and handle potential changes in window_size
        loaded_history = state.get('success_history', [])
        if not isinstance(loaded_history, list):
            loaded_history = list(loaded_history)
        
        self.success_history = loaded_history
        # Truncate or pad history if window_size has changed since checkpoint
        if len(self.success_history) > self.window_size:
            self.success_history = self.success_history[-self.window_size:]
        # elif len(self.success_history) < self.window_size:
            # Potentially pad with neutral values, or just let it repopulate
            # For simplicity, we'll let it repopulate if shorter.

        self.step_count = state.get('step_count', self.step_count)
        
        # Example: if you saved and want to restore base params:
        # self.base_eps = state.get('base_eps', self.base_eps)
        # self.base_lr = state.get('base_lr', self.base_lr)
        # ... and so on for other parameters if saved in get_state ...

        # Ensure success_history is a list after loading (it was deque in some versions)
        if not isinstance(self.success_history, list):
            self.success_history = list(self.success_history) 