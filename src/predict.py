import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids, logits=None, confidence_threshold=0.1):
    """
    Convert BIO tags to spans with optional confidence filtering for PII entities.
    
    Args:
        text: Original text
        offsets: Token offset mappings
        label_ids: Predicted label IDs
        logits: Probability logits (optional, for confidence-based filtering)
        confidence_threshold: Minimum confidence for PII entities (0-1)
    """
    spans = []
    current_label = None
    current_start = None
    current_end = None
    current_confidence = 1.0

    for idx, ((start, end), lid) in enumerate(zip(offsets, label_ids)):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        
        # Calculate confidence if logits provided
        if logits is not None:
            probs = torch.softmax(logits[idx], dim=-1)
            confidence = probs[int(lid)].item()
        else:
            confidence = 1.0
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label, current_confidence))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label, current_confidence))
            current_label = ent_type
            current_start = start
            current_end = end
            current_confidence = confidence
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
                current_confidence = min(current_confidence, confidence)
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label, current_confidence))
                current_label = ent_type
                current_start = start
                current_end = end
                current_confidence = confidence

    if current_label is not None:
        spans.append((current_start, current_end, current_label, current_confidence))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--confidence_threshold", type=float, default=0.5)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids, logits=logits, confidence_threshold=args.confidence_threshold)
            ents = []
            for s, e, lab, conf in spans:
                # Apply confidence threshold filtering for PII entities
                if label_is_pii(lab) and conf < args.confidence_threshold:
                    continue  # Skip low-confidence PII predictions
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()