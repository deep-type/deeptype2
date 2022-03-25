
# constants for saving & loading
# model config:
OBJECTIVE_NAMES = "OBJECTIVE_NAMES"
OBJECTIVE_TYPES = "OBJECTIVE_TYPES"

# inputs:
INPUT_PLACEHOLDERS = "INPUT_PLACEHOLDERS"
LABEL_PLACEHOLDERS = "LABEL_PLACEHOLDERS"
EMBEDDED_INDICES = "EMBEDDED_INDICES"
LABEL_MASK_PLACEHOLDERS = "LABEL_MASK_PLACEHOLDERS"
TRAIN_OP = "TRAIN_OP"
TRAIN_ZERO_ACCUMULATOR_OP = "TRAIN_ZERO_ACCUMULATOR_OP"
TRAIN_ACCUMULATE_GRAD_OP = "TRAIN_ACCUMULATE_GRAD_OP"
TRAIN_ACCUMULATE_OP = "TRAIN_ACCUMULATE_OP"
SEQUENCE_LENGTHS = "SEQUENCE_LENGTHS"
ATTENTION_WEIGHTS = "ATTENTION_WEIGHTS"
IMAGE_SUMMARIES_BW = "IMAGE_SUMMARIES_BW"

GLOBAL_STEP = "global_step"
NLL = "NLL"
NLL_TOTAL = "NLL_TOTAL"
MEMMAP_EMBEDDING_VARIABLES_PATH = "embedding_variables"
CANDIDATE_WORD_DISTANCES = "CANDIDATE_WORD_DISTANCES"
FIELDS_TO_SAVE = [
    "hidden_sizes",
    "transformer_hidden_sizes",
    "transformer_filter_size",
    "n_transformer_heads",
    "objectives",
    "name",
    "cudnn",
    "faux_cudnn",
    "class_weights",
    "features",
    "fused",
    "class_weights_normalize",
    "weight_noise",
    "anneal_rate",
    "anneal_every",
    "macro_anneal_rate",
    "macro_anneal_every",
    "feature_index2words",
    "solver",
    "lr",
    "seed",
    "freeze_rate",
    "freeze_rate_anneal",
    "clip_norm",
    "keep_prob",
    "input_keep_prob",
    "class_weights_clipval",
    "convolutions",
    "create_embedding_lookup",
    "memmap_embedding_variables_path",
    "post_process_spec",
    "gradient_accumulation_steps"
]
EPHEMERAL_FIELDS = ["trainable", "create_variables", "model_load_path", "classifications"]
MEMMAP_FIELDS = [
    "create_embedding_lookup",
    "memmap_embedding_variables_path",
]
OVERRIDEABLE_FIELDS = [
    "keep_prob",
    "name",
    "lr",
    "clip_norm",
    "class_weights_normalize",
    "class_weights_clipval",
    "cudnn",
    "faux_cudnn",
    "anneal_rate",
    "anneal_every",
    "macro_anneal_rate",
    "macro_anneal_every",
    "weight_noise",
    "input_keep_prob",
    "create_embedding_lookup",
    "memmap_embedding_variables_path",
    "post_process_spec",
    "gradient_accumulation_steps"
]
SCENARIO_FEATURE_SCORES = "SCENARIO_FEATURE_SCORES"
FEATURE_INTERACTIONS = "feature_interactions"
