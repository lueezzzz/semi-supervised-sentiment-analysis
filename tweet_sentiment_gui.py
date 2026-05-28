import re
import threading
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import font as tkfont

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError as exc:
    torch = None
    nn = None
    AutoModel = None
    AutoTokenizer = None
    DEPENDENCY_ERROR = exc
else:
    DEPENDENCY_ERROR = None


BASE_DIR = Path(__file__).resolve().parent
EXPORT_DIR = BASE_DIR / "models"


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    architecture: str
    role: str
    priority: int
    model_name: str
    weight_candidates: Tuple[str, ...]
    default_backbone_attr: str
    num_classes: int = 3
    max_length: int = 128


MODEL_CONFIGS: Tuple[ModelConfig, ...] = (
    ModelConfig(
        display_name="XLM-Roberta Teacher (2022)",
        architecture="xlmr",
        role="teacher",
        priority=0,
        model_name="xlm-roberta-base",
        weight_candidates=(
            "xlmr_teacher_2022_model.pt",
            "xlmr_2022_model.pt",
            "xlmr_teacher_model.pt",
            "xlmr_pure_model.pt",
        ),
        default_backbone_attr="xlm_roberta",
    ),
    ModelConfig(
        display_name="XLM-Roberta Student (2025)",
        architecture="xlmr",
        role="student",
        priority=0,
        model_name="xlm-roberta-base",
        weight_candidates=(
            "xlmr_student_2025_model.pt",
            "xlmr_2025_model.pt",
            "xlmr_student_model.pt",
            "xlmr_pseudo_label_model.pt",
        ),
        default_backbone_attr="xlm_roberta",
    ),
    ModelConfig(
        display_name="mBERT Teacher (2022)",
        architecture="mbert",
        role="teacher",
        priority=2,
        model_name="bert-base-multilingual-uncased",
        weight_candidates=(
            "mbert_teacher_2022_model.pt",
            "mbert_2022.pth",
            "mbert_teacher_model.pt",
            "mbert_pure_model.pt",
        ),
        default_backbone_attr="bert",
    ),
    ModelConfig(
        display_name="mBERT Student (2025)",
        architecture="mbert",
        role="student",
        priority=2,
        model_name="bert-base-multilingual-uncased",
        weight_candidates=(
            "mbert_student_2025_model.pt",
            "mbert_2025.pth",
            "mbert_student_model.pt",
            "mbert_pseudo_label_model.pt",
        ),
        default_backbone_attr="bert",
    ),
    ModelConfig(
        display_name="EmoBERT Teacher (2022)",
        architecture="emobert",
        role="teacher",
        priority=1,
        model_name="bhadresh-savani/bert-base-uncased-emotion",
        weight_candidates=(
            "emobert_teacher_2022_model.pt",
            "emobert_2022_model.pt",
            "emobert_teacher_model.pt",
            "emobert_model.pt",

        ),
        default_backbone_attr="bert",
    ),
    ModelConfig(
        display_name="EmoBERT Student (2025)",
        architecture="emobert",
        role="student",
        priority=1,
        model_name="bhadresh-savani/bert-base-uncased-emotion",
        weight_candidates=(
            "emobert_student_2025_model.pt",
            "emobert_2025_model.pt",
            "emobert_student_model.pt",
            "emobert_pseudo_label_model.pt",
            "emobert_2025_pseudo_model.pt",
        ),
        default_backbone_attr="bert",
    ),
)


LABEL_MAPPING: Dict[int, str] = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}

LABEL_COLORS: Dict[str, str] = {
    "Negative": "#c2410c",
    "Neutral": "#475569",
    "Positive": "#15803d",
}


def preprocess_tweet(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = text.replace("#", "")
    return text.lower().strip()


if torch is not None:

    class PureTransformerClassifier(nn.Module):
        def __init__(
            self,
            model_name: str,
            num_classes: int,
            backbone_attr: str,
            state_dict_keys: List[str],
        ):
            super().__init__()
            self.backbone_attr = backbone_attr
            backbone = AutoModel.from_pretrained(model_name)
            setattr(self, backbone_attr, backbone)
            hidden_size = backbone.config.hidden_size
            self.dropout = nn.Dropout(0.2)
            self.classifier = self._build_classifier(hidden_size, num_classes, state_dict_keys)

        @staticmethod
        def _build_classifier(hidden_size: int, num_classes: int, keys: List[str]) -> nn.Module:
            if "classifier.weight" in keys:
                return nn.Linear(hidden_size, num_classes)

            if "classifier.3.weight" in keys:
                return nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes),
                )

            return nn.Linear(hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
            backbone = getattr(self, self.backbone_attr)
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            cls_output = self.dropout(cls_output)
            return self.classifier(cls_output)


@dataclass
class PredictionResult:
    model_name: str
    label: Optional[str] = None
    confidence: Optional[float] = None
    error: Optional[str] = None


class SentimentModelRunner:
    def __init__(self, config: ModelConfig, device):
        self.config = config
        self.device = device
        self.weight_path = self._resolve_weight_path()
        self.tokenizer = None
        self.model = None
        self.loaded = False

    def _resolve_weight_path(self) -> Optional[Path]:
        for filename in self.config.weight_candidates:
            direct_paths = (EXPORT_DIR / filename, BASE_DIR / filename)
            for path in direct_paths:
                if path.exists():
                    return path

        for filename in self.config.weight_candidates:
            matches = list(BASE_DIR.rglob(filename))
            if matches:
                return matches[0]

        return None

    @staticmethod
    def _torch_load(path: Path, device):
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=device)

    @staticmethod
    def _extract_state_dict(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    @staticmethod
    def _strip_module_prefix(state_dict):
        if not any(key.startswith("module.") for key in state_dict.keys()):
            return state_dict
        return {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in state_dict.items()
        }

    def _infer_backbone_attr(self, keys: List[str]) -> str:
        for key in keys:
            marker = ".embeddings."
            if marker in key:
                return key.split(marker, 1)[0]
        return self.config.default_backbone_attr

    def load(self):
        if self.loaded:
            return

        if DEPENDENCY_ERROR is not None:
            raise RuntimeError(
                "Missing Python dependency. Install torch and transformers in this environment."
            ) from DEPENDENCY_ERROR

        if self.weight_path is None:
            candidates = ", ".join(self.config.weight_candidates)
            raise FileNotFoundError(f"Missing model file. Expected one of: {candidates}")

        checkpoint = self._torch_load(self.weight_path, self.device)
        state_dict = self._strip_module_prefix(self._extract_state_dict(checkpoint))
        keys = list(state_dict.keys())
        backbone_attr = self._infer_backbone_attr(keys)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = PureTransformerClassifier(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            backbone_attr=backbone_attr,
            state_dict_keys=keys,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def predict(self, raw_tweet: str) -> PredictionResult:
        try:
            self.load()
            cleaned_tweet = preprocess_tweet(raw_tweet)
            encoding = self.tokenizer(
                cleaned_tweet,
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(logits, dim=1).item()
                confidence = probabilities[0][pred_class].item()

            return PredictionResult(
                model_name=self.config.display_name,
                label=LABEL_MAPPING.get(pred_class, f"Class {pred_class}"),
                confidence=confidence,
            )
        except Exception as exc:
            return PredictionResult(
                model_name=self.config.display_name,
                error=f"{exc.__class__.__name__}: {exc}",
            )


class SentimentGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Philippine Election Tweets Sentiment Analysis")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.device = torch.device("cpu")
        self.runners = [SentimentModelRunner(config, self.device) for config in MODEL_CONFIGS]
        self.config_by_name = {config.display_name: config for config in MODEL_CONFIGS}
        self.result_widgets = {}
        self.is_running = False

        self._configure_styles()
        self._build_layout()

    @staticmethod
    def _emoji_capable_font(size: int = 11):
        available_fonts = set(tkfont.families())
        for family in ("Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", "Apple Color Emoji", "Segoe UI"):
            if family in available_fonts:
                return (family, size)
        return ("Segoe UI", size)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f8fafc")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Title.TLabel", background="#f8fafc", foreground="#0f172a", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#f8fafc", foreground="#475569", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#f8fafc", foreground="#0f172a", font=("Segoe UI", 11, "bold"))
        style.configure("CardTitle.TLabel", background="#ffffff", foreground="#0f172a", font=("Segoe UI", 13, "bold"))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Segoe UI", 9))
        style.configure("Status.TLabel", background="#f8fafc", foreground="#334155", font=("Segoe UI", 9))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 8))

    def _build_layout(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="Philippine Election Related Tweets Sentiment Analysis",
            style="Title.TLabel",
        )
        title.pack(anchor="w")

        subtitle = ttk.Label(
            outer,
            text="Enter one tweet and compare teacher 2022 and student 2025 predictions from XLM-Roberta, mBERT, and EmoBERT.",
            style="Subtitle.TLabel",
        )
        subtitle.pack(anchor="w", pady=(4, 18))

        input_header = ttk.Label(outer, text="Input Tweet", style="Section.TLabel")
        input_header.pack(anchor="w")

        input_panel = ttk.Frame(outer, style="Panel.TFrame", padding=12)
        input_panel.pack(fill="x", pady=(6, 14))

        self.tweet_text = tk.Text(
            input_panel,
            height=6,
            wrap="word",
            font=self._emoji_capable_font(11),
            relief="flat",
            background="#ffffff",
            foreground="#0f172a",
            insertbackground="#0f172a",
        )
        self.tweet_text.pack(fill="x", expand=False)
        self.tweet_text.insert("1.0", "Sana talaga manalo si Leni sa darating na eleksyon! #LeniRobredo2022")

        button_row = ttk.Frame(outer)
        button_row.pack(fill="x", pady=(0, 18))

        self.clear_button = ttk.Button(button_row, text="Clear", command=self.clear_input)
        self.clear_button.pack(side="left")

        self.classify_button = ttk.Button(
            button_row,
            text="Classify Tweet",
            style="Accent.TButton",
            command=self.classify_tweet,
        )
        self.classify_button.pack(side="right")

        final_header = ttk.Label(outer, text="Final Sentiment", style="Section.TLabel")
        final_header.pack(anchor="w")

        final_panel = ttk.Frame(outer, style="Panel.TFrame", padding=16)
        final_panel.pack(fill="x", pady=(6, 14))

        self.final_label = tk.Label(
            final_panel,
            text="Waiting",
            bg="#ffffff",
            fg="#475569",
            font=("Segoe UI", 24, "bold"),
            anchor="w",
        )
        self.final_label.pack(anchor="w")

        self.final_detail = ttk.Label(
            final_panel,
            text="The final sentiment will be computed from the six model votes.",
            style="Muted.TLabel",
            wraplength=1050,
        )
        self.final_detail.pack(anchor="w", pady=(6, 0))

        result_header = ttk.Label(outer, text="Model Classifications", style="Section.TLabel")
        result_header.pack(anchor="w")

        cards = ttk.Frame(outer)
        cards.pack(fill="both", expand=True, pady=(6, 12))
        card_columns = 3
        cards.columnconfigure(tuple(range(card_columns)), weight=1, uniform="result_cards")
        cards.rowconfigure((0, 1), weight=1, uniform="result_rows")

        for index, config in enumerate(MODEL_CONFIGS):
            row = index // card_columns
            column = index % card_columns
            card = ttk.Frame(cards, style="Panel.TFrame", padding=16)
            card.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=(0 if column == 0 else 8, 0 if column == card_columns - 1 else 8),
                pady=(0 if row == 0 else 8, 8 if row == 0 else 0),
            )

            title = ttk.Label(card, text=config.display_name, style="CardTitle.TLabel")
            title.pack(anchor="w")

            label = tk.Label(
                card,
                text="Waiting",
                bg="#ffffff",
                fg="#475569",
                font=("Segoe UI", 18, "bold"),
                anchor="w",
            )
            label.pack(anchor="w", pady=(16, 2))

            confidence = ttk.Label(card, text="Confidence: --", style="Muted.TLabel")
            confidence.pack(anchor="w")

            bar = ttk.Progressbar(card, orient="horizontal", mode="determinate", maximum=100, value=0)
            bar.pack(fill="x", pady=(14, 10))

            detail = ttk.Label(card, text="Model will load on first classification.", style="Muted.TLabel", wraplength=280)
            detail.pack(anchor="w")

            self.result_widgets[config.display_name] = {
                "label": label,
                "confidence": confidence,
                "bar": bar,
                "detail": detail,
            }

        self.status = ttk.Label(
            outer,
            text=f"Ready. Device: {self.device}",
            style="Status.TLabel",
        )
        self.status.pack(anchor="w", side="bottom")

    def clear_input(self):
        if self.is_running:
            return
        self.tweet_text.delete("1.0", "end")
        self._set_final_waiting()
        for config in MODEL_CONFIGS:
            self._set_waiting(config.display_name)
        self.status.config(text=f"Ready. Device: {self.device}")

    def classify_tweet(self):
        if self.is_running:
            return

        raw_tweet = self.tweet_text.get("1.0", "end").strip()
        if not raw_tweet:
            messagebox.showinfo("No tweet entered", "Please enter a tweet before classifying.")
            return

        self.is_running = True
        self.classify_button.config(state="disabled")
        self.clear_button.config(state="disabled")
        self.status.config(text="Classifying tweet. Models may take a moment to load the first time.")
        self._set_final_loading()

        for config in MODEL_CONFIGS:
            self._set_loading(config.display_name)

        worker = threading.Thread(target=self._run_predictions, args=(raw_tweet,), daemon=True)
        worker.start()

    def _run_predictions(self, raw_tweet: str):
        results = []
        try:
            for runner in self.runners:
                results.append(runner.predict(raw_tweet))
        except Exception:
            results.append(
                PredictionResult(
                    model_name="Application",
                    error=traceback.format_exc(limit=2),
                )
            )
        self.root.after(0, self._show_results, results)

    def _show_results(self, results: List[PredictionResult]):
        for result in results:
            if result.model_name not in self.result_widgets:
                continue

            if result.error:
                self._set_error(result.model_name, result.error)
            else:
                self._set_prediction(result.model_name, result.label, result.confidence)

        final_label, final_detail = self._calculate_final_sentiment(results)
        self._set_final_prediction(final_label, final_detail)

        self.is_running = False
        self.classify_button.config(state="normal")
        self.clear_button.config(state="normal")
        self.status.config(text=f"Done. Device: {self.device}")

    def _calculate_final_sentiment(self, results: List[PredictionResult]):
        successful_results = [
            result for result in results
            if result.label and not result.error and result.model_name in self.config_by_name
        ]
        if not successful_results:
            return "Unavailable", "No model returned a successful prediction."

        vote_counts = Counter(result.label for result in successful_results)
        highest_votes = max(vote_counts.values())
        tied_labels = [label for label, count in vote_counts.items() if count == highest_votes]
        vote_summary = ", ".join(f"{label}: {vote_counts[label]}" for label in LABEL_MAPPING.values())

        if len(tied_labels) == 1:
            label = tied_labels[0]
            return label, f"Majority vote selected {label}. Votes: {vote_summary}."

        student_results = [
            result for result in successful_results
            if self.config_by_name[result.model_name].role == "student" and result.label in tied_labels
        ]
        student_counts = Counter(result.label for result in student_results)
        if student_counts:
            highest_student_votes = max(student_counts.values())
            student_tied_labels = [
                label for label in tied_labels
                if student_counts.get(label, 0) == highest_student_votes
            ]
            if len(student_tied_labels) == 1:
                label = student_tied_labels[0]
                return (
                    label,
                    f"Overall vote tied, so student votes broke the tie in favor of {label}. Votes: {vote_summary}.",
                )
            tied_labels = student_tied_labels

        priority_results = sorted(
            student_results,
            key=lambda result: self.config_by_name[result.model_name].priority,
        )
        for result in priority_results:
            if result.label in tied_labels:
                config = self.config_by_name[result.model_name]
                label = result.label
                return (
                    label,
                    "Student votes were tied, so priority resolved it: "
                    f"XLM-Roberta > EmoBERT > mBERT. Selected {label} from {config.display_name}. "
                    f"Votes: {vote_summary}.",
                )

        label = tied_labels[0]
        return label, f"Vote tie remained after tiebreakers. Selected {label}. Votes: {vote_summary}."

    def _set_waiting(self, model_name: str):
        widgets = self.result_widgets[model_name]
        widgets["label"].config(text="Waiting", fg="#475569")
        widgets["confidence"].config(text="Confidence: --")
        widgets["bar"].config(value=0)
        widgets["detail"].config(text="Model will load on first classification.")

    def _set_final_waiting(self):
        self.final_label.config(text="Waiting", fg="#475569")
        self.final_detail.config(text="The final sentiment will be computed from the six model votes.")

    def _set_final_loading(self):
        self.final_label.config(text="Classifying", fg="#2563eb")
        self.final_detail.config(text="Waiting for model predictions before applying majority voting.")

    def _set_final_prediction(self, label: str, detail: str):
        self.final_label.config(text=label, fg=LABEL_COLORS.get(label, "#0f172a"))
        self.final_detail.config(text=detail)

    def _set_loading(self, model_name: str):
        widgets = self.result_widgets[model_name]
        widgets["label"].config(text="Loading", fg="#2563eb")
        widgets["confidence"].config(text="Confidence: --")
        widgets["bar"].config(value=0)
        widgets["detail"].config(text="Preparing tokenizer and model weights.")

    def _set_prediction(self, model_name: str, label: str, confidence: float):
        percent = confidence * 100
        widgets = self.result_widgets[model_name]
        widgets["label"].config(text=label, fg=LABEL_COLORS.get(label, "#0f172a"))
        widgets["confidence"].config(text=f"Confidence: {percent:.2f}%")
        widgets["bar"].config(value=percent)
        widgets["detail"].config(text="Classification completed successfully.")

    def _set_error(self, model_name: str, error: str):
        short_error = error if len(error) <= 180 else error[:177] + "..."
        widgets = self.result_widgets[model_name]
        widgets["label"].config(text="Unavailable", fg="#b91c1c")
        widgets["confidence"].config(text="Confidence: --")
        widgets["bar"].config(value=0)
        widgets["detail"].config(text=short_error)


def main():
    root = tk.Tk()
    app = SentimentGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
