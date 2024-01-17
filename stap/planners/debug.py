import torch


def debug_value_fn(state, action_space, value_fn, decode_fn):
    obs = decode_fn(state)
    # Generate a list of 1D tensors representing the linspace for each dimension
    linspace_list = [
        torch.linspace(action_space.low[dim], action_space.high[dim], 10, device=state.device)
        for dim in range(action_space.shape[0])
    ]

    # Generate the meshgrid using the lists of linspace tensors
    grids = torch.meshgrid(*linspace_list)

    # Stack the grids into a 2D tensor, where the second dimension is action_space.shape[0]
    # and each row is a unique combination of actions
    action_combinations = torch.stack([grid.flatten() for grid in grids], dim=-1)

    # Create a batch of states
    states = torch.stack([obs] * action_combinations.shape[0], dim=0)

    values = torch.mean(torch.stack(value_fn(states, action_combinations), dim=1), dim=1)
    stop = 0
