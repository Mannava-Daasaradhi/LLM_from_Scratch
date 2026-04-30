import yaml
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float

@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    warmup_steps: int
    max_steps: int
    eval_interval: int
    eval_steps: int
    checkpoint_dir: str
    log_interval: int

@dataclass
class DataConfig:
    train_file: str
    val_file: str
    tokenizer_path: str

@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    device: str

def load_config(path: str) -> Config:
    """Loads the YAML configuration file and parses it into a strongly typed Config object."""
    with open(path, 'r', encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Unpack the nested dictionaries directly into their respective dataclasses
    return Config(
        model=ModelConfig(**raw_config['model']),
        training=TrainingConfig(**raw_config['training']),
        data=DataConfig(**raw_config['data']),
        device=raw_config.get('device', 'cuda') # defaults to cuda if missing
    )

# --- Quick Test ---
if __name__ == "__main__":
    # Assuming config.py is in the root and running it from there
    cfg = load_config("configs/shakespeare.yaml")
    print(f"Loaded config successfully!")
    print(f"Vocab Size: {cfg.model.vocab_size}")
    print(f"Learning Rate: {cfg.training.learning_rate}")
    print(f"Device: {cfg.device}")