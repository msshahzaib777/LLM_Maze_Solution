import torch
import matplotlib.pyplot as plt
import os, re
import constant as config
from constant import ADAPTER_DIR
from datetime import datetime

adaptername = config.ADAPTER_NAME
EVAL_STEPS = config.EVAL_STEPS

def load_training_state(adapter_name):
    # Find the latest checkpoint with training_state.pt
    adapter_path = f"{ADAPTER_DIR}/{adapter_name}"
    checkpoint_dirs = [d for d in os.listdir(adapter_path) if d.startswith('step_')]
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split('_')[1]))[-1]
    print("loading from ", latest_checkpoint)
    # latest_checkpoint = str(10400)  # Manually set for now
    state_path = os.path.join(adapter_path, latest_checkpoint, 'training_state.pt')
    state = torch.load(state_path, map_location='cpu')
    print("loaded state keys: ", state.keys())
    print("training steps: ", len(state.get('train_losses', [])))
    print("eval steps: ", len(state.get('eval_losses', [])))  
    print("loss steps: ", len(state.get('loss_steps', []))) 
    train_losses = state.get('train_losses', [])
    eval_losses = state.get('eval_losses', [])
    loss_steps = state.get('loss_steps', [])

    # Manually align eval_steps: spread them evenly across the range of loss_steps
    if len(eval_losses) > 0 and len(loss_steps) > 0:
        start, end = loss_steps[0], loss_steps[-1]
        if len(eval_losses) == 1:
            eval_steps = [start]
        else:
            eval_steps = [int(start + i * (end - start) / (len(eval_losses) - 1)) for i in range(len(eval_losses))]
    else:
        eval_steps = list(range(config.EVAL_STEPS, len(eval_losses) * config.EVAL_STEPS + 1, config.EVAL_STEPS))

    return list(zip(loss_steps, train_losses)), list(zip(eval_steps, eval_losses))

def plot_losses(train_data, eval_data, adapter_name):
    train_steps, train_losses = zip(*train_data)
    eval_steps, eval_losses = zip(*eval_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_losses, label='Training Loss')
    plt.plot(eval_steps, eval_losses, label='Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./plots/{adapter_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

def extract_losses(log_files, avg_steps=100):
    train, eval, buf, steps = {}, {}, [], []
    for log in log_files:
        with open(log) as f:
            for l in f:
                m1 = re.search(r"Step (\d+).*Loss: ([\d\.eE+-]+)", l)
                if m1:
                    steps.append(int(m1.group(1)))
                    buf.append(float(m1.group(2)))
                    if len(buf) == avg_steps:
                        train[steps[-1]] = sum(buf) / avg_steps
                        buf, steps = [], []
                m2 = re.search(r"Validation loss at step (\d+): ([\d\.eE+-]+)", l)
                if m2:
                    eval[int(m2.group(1))] = float(m2.group(2))
    if buf:
        train[steps[-1]] = sum(buf) / len(buf)
    return train, eval

# log_dir = "./logs"
# log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith(f"{adaptername}_") and f.endswith(".log")]
# train, eval = extract_losses(log_files)
# plot_losses(sorted(train.items()), sorted(eval.items()), adaptername)
# train_data, eval_data = load_training_state(adaptername)
# plot_losses(train_data, eval_data, adaptername)