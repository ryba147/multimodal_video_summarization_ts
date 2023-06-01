import os

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# summe_base_path: str = "SumMe/"
# tvsum_base_path: str = "tvsum50_ver_1_1/ydata-tvsum50-v1_1/"

DATALOADER_BATCH_SIZE: int = 32
N_HEADS: int = 4
LEARNING_RATE: float = 0.00005
EPOCHS: int = 200
HIDDEN_DIM: int = 512
VISUAL_INPUT_DIM: int = 1024

SUMME_SPLIT = "data/splits/summe.json"
TVSUM_SPLIT = "data/splits/tvsum.json"
