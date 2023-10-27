import torch
import logging

def set_device(device_id: int = 0) -> torch.device:
    """
    Check cuda is available or not
    """
    logging.info("Check cuda is available or not:")

    # Check cuda in device
    if torch.cuda.is_available():
        if device_id != -1:
            torch.cuda.set_per_process_memory_fraction(0.5, device_id)
            device = torch.device(f"cuda:{device_id}")
            logging.info(torch.cuda.get_device_name())
            logging.info("Cuda is available")
        else:
            device = torch.device("cpu")
            logging.info("Cuda is not used")
    else:
        device = torch.device("cpu")
        logging.info("Cuda is not available")
    return device