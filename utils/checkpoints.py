## Checkpoint function
import torch

def checkpoint(model, file_path):
    """
    Save state dict models in process train
    """
    torch.save(model.state_dict(), file_path)