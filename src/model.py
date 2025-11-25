from transformers import AutoModelForTokenClassification, AutoConfig
from labels import LABEL2ID, ID2LABEL


def create_model(model_name: str):
    """
    Create a token classification model with improved configuration for PII detection.
    
    Args:
        model_name: HuggingFace model identifier
    
    Returns:
        Configured AutoModelForTokenClassification
    """
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(LABEL2ID)
    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    
    model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID,
)
    return model