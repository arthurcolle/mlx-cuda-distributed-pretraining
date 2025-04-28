import re
import pandas as pd
import matplotlib.pyplot as plt

def parse_log(file_path):
    train_pattern = re.compile(
        r"Step (?P<step>\d+): loss=(?P<loss>\d+\.\d+) \| ppl=(?P<ppl>\d+\.\d+) \| lr=(?P<lr>[\d\.e\-]+).*?tokens_per_sec=(?P<tokens_per_sec>\d+\.\d+)"
    )
    val_pattern = re.compile(r"val_loss=(?P<val_loss>\d+\.\d+) \| val_ppl=(?P<val_ppl>\d+\.\d+)")
    train_data = []
    val_data = []
    with open(file_path, "r") as f:
        for line in f:
            t = train_pattern.search(line)
            if t:
                train_data.append({
                    "step": int(t.group("step")),
                    "loss": float(t.group("loss")),
                    "ppl": float(t.group("ppl")),
                    "lr": float(t.group("lr")),
                    "tokens_per_sec": float(t.group("tokens_per_sec")),
                })
            v = val_pattern.search(line)
            if v:
                val_data.append({
                    "val_loss": float(v.group("val_loss")),
                    "val_ppl": float(v.group("val_ppl")),
                })
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    return train_df, val_df

def plot_curves(train_df, val_df):
    plt.figure()
    plt.plot(train_df["step"], train_df["loss"])
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.show()

    plt.figure()
    plt.plot(train_df["step"], train_df["ppl"])
    plt.xlabel("Step")
    plt.ylabel("Training Perplexity")
    plt.title("Training PPL Curve")
    plt.show()

    if not val_df.empty:
        plt.figure()
        plt.plot(range(len(val_df)), val_df["val_loss"])
        plt.xlabel("Validation Checkpoint")
        plt.ylabel("Validation Loss")
        plt.title("Validation Loss Curve")
        plt.show()

if __name__ == "__main__":
    import sys
    log_path = sys.argv[1] if len(sys.argv) > 1 else "training.log"
    train_df, val_df = parse_log(log_path)
    plot_curves(train_df, val_df)

# Save this script as `plot_training_curves.py` and run:
# python plot_training_curves.py path/to/your/logfile.log
