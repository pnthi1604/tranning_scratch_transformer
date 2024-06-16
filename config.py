import torch
from .utils.folders import join_base
from .utils.optimizers import (
    ADAMW,
    RADAM,
)
from .models.utils import (
    SEQ2SEQ_TRANSFORMER,
)

def get_config(base_dir: str=None):
    config = {}

    if not base_dir:
        config["base_dir"] = "./"
    else:
        config["base_dir"] = base_dir

    # Tokenizer
    config['tokenizer_tgt_path'] = None
    config['tokenizer_src_path'] = None
    config["use_tokenizer"] = "huggingface"
    config["special_tokens"] = [
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>"
    ]
    config["vocab_size"] = 30000
    config['min_frequency'] = 2

    # Directories
    config["model_folder_name"] = join_base(config["base_dir"], "/model")
    config["model_folder_name_zip"] = join_base(config["base_dir"], "/model.zip")
    config["model_base_name"] = "model_"
    config["model_out"] = "out_"
    
    config["preload"] = "latest"
    config["data"] = join_base(config["base_dir"], "/data")
    config["log_dir"] = join_base(config["base_dir"], "/log")
    config["log_dir_zip"] = join_base(config["base_dir"], "/log.zip")
    config["log_files"] = join_base(config["log_dir"], "/log_files")
    config["loss_train"] = join_base(config["log_files"], "/loss_train.json")
    config["loss_val"] = join_base(config["log_files"], "/loss_val.json")
    config["loss_train_step"] = join_base(config["log_files"], "/loss_train_step.json")
    config["loss_val_step"] = join_base(config["log_files"], "/loss_val_step.json")
    config["learning_rate_step"] = join_base(config["log_files"], "/learning_rate_step.json")
    config["timestep_train"] = join_base(config["log_files"], "/timestep_train.json")
    config["timestep_val"] = join_base(config["log_files"], "/timestep_val.json")
    config["timestep_train_and_val"] = join_base(config["log_files"], "/timestep_train_and_val.json")
    config["timestep_lr"] = join_base(config["log_files"], "/timestep_lr.json")

    # Dataset
    config["lang_src"] = "noise_vi"
    config["lang_tgt"] = "vi"
    config["train_ds_path"] = None
    config["val_ds_path"] = None
    config["test_ds_path"] = None
    config["max_num_test"] = 2000
    config["max_num_val"] = 10000
    config["corpus"] = None
    config["max_len"] = 100
    config["src_vocab_size_bart_encoder"] = 30000
    config["seed"] = 42

    # Train
    config["model_name"] = SEQ2SEQ_TRANSFORMER
    config["step_train"] = None
    config["preload"] = "latest"
    config["pretrain"] = False
    config["continue_step"] = False
    
    # Trainning loop
    config["big_batch"] = 128
    config["batch_train"] = 32
    config["batch_val"] = 32
    config["batch_test"] = 1
    config["num_steps"] = 15000
    config["val_steps"] = config["num_steps"] // 5

    # config model
    config["d_model"] = 768
    config["encoder_layers"] = 6
    config["decoder_layers"] = 6
    config["encoder_attention_heads"] = 12
    config["decoder_attention_heads"] = 12
    config["decoder_ffn_dim"] = 3072
    config["encoder_ffn_dim"] = 3072
    config["activation_function"] = "gelu"
    config["dropout"] = 0.1
    config["attention_dropout"] = 0.1
    config["activation_dropout"] = 0.1
    config["classifier_dropout"] = 0.0
    config["max_position_embeddings"] = config["max_len"] # The maximum sequence length
    config["init_std"] = 0.02 # Std for initializing all weight matrices
    config["encoder_layerdrop"] = 0.0 # Dropout encoder layer
    config["decoder_layerdrop"] = 0.0 # Dropout decoder layer
    config["scale_embedding"] = False # Scale embeddings with sqrt(d_model)
    config["num_beams"] = 1
    config["init_type"] = "normal"
    config["share_tgt_emb_and_out"] = False
    config["checkpoint"] = None
    config["num_labels"] = None

    # Random Encoder config
    config["random_encoder_layers"] = 2
    config["random_decoder_layers"] = 2
    config["random_encoder_attention_heads"] = 4
    config["random_decoder_attention_heads"] = 4
    config["random_decoder_ffn_dim"] = 512
    config["random_encoder_ffn_dim"] = 512
    config["random_activation_function"] = "gelu"
    config["random_dropout"] = 0.1
    config["random_attention_dropout"] = 0.1
    config["random_activation_dropout"] = 0.1

    # Optimizer
    config["optimizer_name"] = ADAMW
    config["weight_decay"] = 0.0
    config["lr"] = 0.5
    config["eps"] = 1e-9
    config["betas"] = (0.9, 0.98)

    # CrossEntropyLoss
    config["label_smoothing"] = 0.01

    # Scheduler (Noam decay)
    config["warmup_steps"] = 4000

    # Device
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Metric
    config["f_beta"] = 0.5
    config["beams"] = [1]
    config["use_pytorch_metric"] = False
    config["use_bleu"] = True
    config["use_recall"] = False
    config["use_precision"] = False
    config["use_rouge"] = False

    return config