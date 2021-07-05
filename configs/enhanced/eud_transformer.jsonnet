local transformer_model = std.extVar("MODEL_NAME");
local max_length = 128;
local do_lower_case = std.extVar("DO_LOWER_CASE");
local transformer_dim = 768;
local encoder_dim = transformer_dim;

local tbid = std.extVar("TBID");
local treebank = std.extVar("TREEBANK");

local feedforward_common = {
    "activations": "elu",
    "dropout": 0.33,
    "hidden_dims": 200,
    "input_dim": encoder_dim,
    "num_layers": 1
};

{
    "dataset_reader": {
        "type": "universal_dependencies_enhanced",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model,
                "max_length": max_length,
                "tokenizer_kwargs": {
                    "do_lower_case": false,
                    "tokenize_chinese_chars": true,
                    "strip_accents": false,
                    "clean_text": true,
                }   
            }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    "model": {
        "type": "enhanced_kg_parser",
        "word_embedder": {
            "token_embedders": {          
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": transformer_model,
                    "max_length": max_length
                }
            }
        },
        "encoder_dim": encoder_dim,
        "tag_representation_dim": 300,
        "arc_representation_dim": 300,
        "dropout": 0.35,
        "input_dropout": 0.35,
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "sorting_keys": ["words"],
            "batch_size" : 16
        }
    },

  "trainer": {
    "num_epochs": 50,
    "grad_norm": 5.0,
    "patience": 10,
    "validation_metric": "+labeled_f1",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 3e-4,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
  },
  "random_seed": std.parseInt(std.extVar("RANDOM_SEED")),
  "numpy_seed": std.parseInt(std.extVar("NUMPY_SEED")),
  "pytorch_seed": std.parseInt(std.extVar("PYTORCH_SEED")),
}
