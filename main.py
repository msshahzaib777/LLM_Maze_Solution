import argparse
import types
import time
from datasets import load_dataset
import mlx as mx
from mlx.utils import tree_map
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingCallback
from mlx_lm.lora import run

folder="output/"

system_message = """<Your System Message>"""
def create_conversation(sample):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": sample["text"]},
      {"role": "assistant", "content": sample["diagnosis"]}
    ]
  }
dataset = load_dataset("<Your Datacard>", split="train")
dataset = dataset.shuffle()
dataset = dataset.map(create_conversation, batched=False)
dataset = dataset.train_test_split(test_size=10/90)
dataset_test_valid = dataset['test'].train_test_split(test_size=20/80)
dataset["train"].to_json(folder + "train.jsonl", orient="records")
dataset_test_valid["train"].to_json(folder + "test.jsonl", orient="records")
dataset_test_valid["test"].to_json(folder + "valid.jsonl", orient="records")

# mx.distributed will automatically discover, use MPI.
world = mx.distributed.init()
size = world.size()


def all_reduce_grads(grads):
    # tree_map is walking through the full structure of gradients and applying the all_sum / size logic to each individual tensor inside.
    return tree_map(lambda x: mx.distributed.all_sum(x) / size, grads)


# vanilla version
class GradientAveragingCallback(TrainingCallback):

    def __init__(self):
        pass  # No need to initialize loss tracking anymore

    # This runs after backwards pass but before optimizer step
    def on_after_backward(self, model, grads, step):
        new_grads = all_reduce_grads(grads)
        return new_grads

    # Empty implementations of the reporting methods
    def on_train_loss_report(self, info):
        pass  # No tracking or reporting of train loss

    # Empty implementations of the reporting methods
    def on_val_loss_report(self, info):
        pass  # No tracking or reporting of validation loss


def main():
    print(f"Running in distributed mode: Process {world.rank()} of {size} - synchronizing gradients across processes.",flush=True)

    parser = argparse.ArgumentParser(
        description="Fine-tune language models using MLX with optional LoRA adapters.")
    parser.add_argument("--model", type=str, default="ministral/Ministral-3b-instruct",
                        help="Model identifier or local path.")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--data", type=str, default="<Your Data>")
    parser.add_argument("--fine-tune-type", type=str, default="lora")
    parser.add_argument("--num-layers", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10000) # based on your epoch and datasize
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default="adapters")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-batches", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lora-parameters", type=dict,
                        default={
                            "keys": ['mlp.gate_proj', 'mlp.down_proj', 'self_attn.q_proj', 'mlp.up_proj',
                                   'self_attn.o_proj', 'self_attn.v_proj', 'self_attn.k_proj'],
                            "rank": 8,
                            "alpha": 8,
                            "dropout": 0.05,
                            "scale": 16.0})
    parser.add_argument("--lr-schedule", type=str, default=None)

    args = parser.parse_args()

    start_time = time.time()

    # Initialize the model architecture
    model = load(args.model)

    # Initialize gradient synchronization handler
    # This manages gradient averaging across distributed processes
    grad_sync_handler = GradientAveragingCallback()

    # Launch the fine-tuning process
    # The training controller manages batches, optimization steps, and validation
    run(types.SimpleNamespace(**vars(args)),
        training_callback=grad_sync_handler)

    elapsed_time = time.time() - start_time
    print(f"Total training duration: {elapsed_time:.2f} seconds",flush=True)

if __name__ == "__main__":
    main()