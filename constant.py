# Get the project root directory
ROOT_DIR = "/home/mshahz02/fileserverdata/"
MODEL_DIR = ROOT_DIR + "models/huggingface"
ADAPTER_DIR = ROOT_DIR + "models/finetuned/adapter"
MERGED_MODEL_DIR = ROOT_DIR + "models/finetuned/merged_models"
DATA_DIR = ROOT_DIR + "data"

base_id = "Qwen/Qwen3-14B"
# base_id = MERGED_MODEL_DIR + "/qwen3_4B_123_36000"
Dataset_NAME = "env_agent_v1"
# ADAPTER_NAME = "Adapter_qwen34B_123_234"
# ADAPTER_NAME = "qwen34B_123"
ADAPTER_NAME = "paperbased_adapter_env_agent_v1"
adapter_base_path = f"{ADAPTER_DIR}/{ADAPTER_NAME}"
quantized = False
EPOCH = 15
step_num = 75000 #104000 #36000  # Set to specific step number to override automatic selection
BATCH_SIZE = 2  # Increased batch size for better GPU utilization
ACCUM = 64//BATCH_SIZE  # Gradient accumulation steps
MIN_LR = 1e-6
MAX_NEW_TOKENS = 512
INF_MAX_NEW_TOKENS = 2048
test_BATCH_SIZE = 32
EVAL_STEPS = 100
