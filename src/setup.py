import torch
import random
import numpy as np
import logging
from transformers import DistilBertTokenizer

# Set up GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set a random seed for reproducibility
def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

set_seed(42)

# Set up logging for tracking progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Environment prepared and seed set for reproducibility.")
