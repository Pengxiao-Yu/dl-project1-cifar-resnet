"""Create the <=8-page report only from completed, measured experiment artifacts.
Requires a real public-repository URL. Do not run before validation results exist.
"""

import argparse, hashlib, json, re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-url", required=True)
    p.add_argument(
        "--draft", action="store_true", help="Mark proposed repository as unpublished"
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if not re.fullmatch(
        r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", a.repo_url
    ):
        raise ValueError("Supply the verified public GitHub repository URL")
    run = ROOT / "results/main"
    s = json.loads((run / "summary.json").read_text())
    c = json.loads((run / "config.json").read_text())
    h = json.loads((run / "history.json").read_text())
    d = json.loads((run / "diagnostics.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    pred = json.loads((run / "submission.json").read_text())
    assert s["complete"] and s["epochs_completed"] == 80 and len(h) == 80
    assert s["parameters"] == 4368690 and s["parameters"] <= 5_000_000
    assert abs(s["validation"]["accuracy"] - d["accuracy"]) < 1e-10
    assert (
        abs(s["validation"]["accuracy"] - pred["validation_recheck"]["accuracy"])
        < 1e-10
    )
    assert pred["checkpoint_sha256"] == s["checkpoint_sha256"]
    actual_checkpoint_hash = hashlib.sha256((run / "best.pt").read_bytes()).hexdigest()
    assert actual_checkpoint_hash == s["checkpoint_sha256"]
    scorepath = run / "kaggle_result.json"
    kg = json.loads(scorepath.read_text()) if scorepath.exists() else None
    if kg:
        assert (
            kg.get("verified_from_ui") is True
            and kg.get("submission_sha256") == pred["submission_sha256"]
        )
    score = f"{100*float(kg['score']):.2f}%" if kg else "Not yet verified on Kaggle"
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyX",
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            spaceAfter=8,
            textColor=colors.HexColor("#213047"),
        )
    )
    styles.add(
        ParagraphStyle(name="SmallX", parent=styles["BodyX"], fontSize=8, leading=11)
    )
    styles.add(
        ParagraphStyle(
            name="H1X",
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            spaceAfter=14,
            textColor=colors.HexColor("#193C60"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2X",
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=16,
            spaceBefore=6,
            spaceAfter=8,
            textColor=colors.HexColor("#193C60"),
        )
    )
    story = []

    def para(t, sty="BodyX"):
        return Paragraph(t, styles[sty])

    def add(t, sty="BodyX"):
        story.append(para(t, sty))

    def table(rows, widths):
        t = Table(
            [[para(str(x), "SmallX") for x in row] for row in rows],
            colWidths=widths,
            hAlign="LEFT",
            repeatRows=1,
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EFF5")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#A4B2C2")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#D8DFE8")),
                ]
            )
        )
        story.append(t)
        story.append(Spacer(1, 10))

    def figure(name, maxw=490, maxh=300):
        path = run / "figures" / name
        w, hh = ImageReader(str(path)).getSize()
        scale = min(maxw / w, maxh / hh)
        story.append(
            Image(str(path), width=w * scale, height=hh * scale, hAlign="CENTER")
        )

    acc = 100 * s["validation"]["accuracy"]
    best = s["best_epoch"]
    first = h[0]["validation"]["accuracy"] * 100
    minutes = s["elapsed_seconds"] / 60
    if a.draft:
        add("REVIEW DRAFT - publication and submission pending", "SmallX")
    add("A CIFAR Residual Network Under Five Million Parameters", "H1X")
    add(
        "Pengxiao Yu (py2352) | Individual project<br/>Deep Learning, Fall 2026",
        "BodyX",
    )
    add(
        f'{"Proposed repository (not yet published)" if a.draft else "Public code and executed notebook"}:<br/><link href="{a.repo_url}" color="#174D88">{a.repo_url}</link>',
        "BodyX",
    )
    add("1. Overview", "H2X")
    add(
        "This project adapts a basic-block residual network to 32 x 32 color images under a strict model-size budget. It uses a narrow CIFAR stem, four residual stages and global average pooling. All weights are initialized from scratch; the training pipeline uses only images and labels provided in the competition archive."
    )
    add(
        f'The network has <b>{s["parameters"]:,} trainable parameters</b>. In one fixed-seed run of {len(h)} epochs, the selected checkpoint at epoch {best} achieved <b>{acc:.2f}% accuracy</b> on the fixed validation partition. Any verified competition score is reported separately below. A public leaderboard score applies to its scored subset; private-test accuracy requires separate verification.'
    )
    table(
        [
            ["Measured quantity", "Result"],
            ["Trainable parameters", f'{s["parameters"]:,} / 5,000,000'],
            [
                "Training / validation examples",
                f'{manifest["train_count"]:,} / {manifest["val_count"]:,}',
            ],
            [
                "Selected validation accuracy",
                f'{acc:.2f}% ({s["validation"]["correct"]:,} / {s["validation"]["count"]:,})',
            ],
            [
                (
                    kg.get("score_label", "Kaggle leaderboard test accuracy")
                    if kg
                    else "Kaggle score"
                ),
                score,
            ],
            ["Epochs / selected epoch", f"{len(h)} / {best}"],
            [
                "Epoch training + validation time",
                f'{minutes:.2f} min on {escape(s["device"])} (Apple M1 Pro)',
            ],
        ],
        [215, 275],
    )
    add(
        "The budget check, residual-path identity, finite gradients, checkpoint round-trip and augmentation isolation were verified before the experiment. Formal results come from the full data run, while synthetic engineering checks are recorded separately.",
        "BodyX",
    )
    add(
        "AI assistance disclosure: OpenAI Codex assisted implementation, experiment orchestration and report drafting. All numerical results in this report derive from saved executed artifacts.",
        "SmallX",
    )
    story.append(PageBreak())
    add("2. Model and Data Protocol", "H1X")
    add("Residual architecture", "H2X")
    add(
        "Each basic block computes ReLU(S(x) + BN(Conv3x3(ReLU(BN(Conv3x3(x)))))). The skip path is an identity unless spatial size or channels change, when a stride-matched 1 x 1 convolution and batch normalization project the input. This follows the residual-learning formulation of He et al. [1]."
    )
    table(
        [
            ["Stage", "Output shape", "Blocks / operation"],
            ["Input", "3 x 32 x 32", "RGB competition image"],
            ["Stem", "40 x 32 x 32", "3 x 3 conv, stride 1; BN; ReLU"],
            ["Residual stage 1", "40 x 32 x 32", "2 basic blocks"],
            ["Residual stage 2", "80 x 16 x 16", "2 blocks; first stride 2"],
            ["Residual stage 3", "160 x 8 x 8", "2 blocks; first stride 2"],
            ["Residual stage 4", "320 x 4 x 4", "2 blocks; first stride 2"],
            ["Readout", "10 logits", "Global average pooling; linear layer"],
        ],
        [120, 140, 230],
    )
    add("Design tradeoffs", "H2X")
    add(
        "The stride-one stem and absence of an early max pool preserve information at CIFAR resolution. Reducing base width to 40 controls the roughly quadratic cost of channel width while retaining four stages and eight residual blocks. Global average pooling avoids a large flattened classifier. Projection skips add parameters but provide a learned match at resolution transitions. This run evaluates the resulting design; it does not establish that any one choice is independently optimal."
    )
    add("Dataset and separation", "H2X")
    add(
        f'The instructor archive contains {manifest["train_shape"][0]:,} labeled training images and {manifest["test_shape"][0]:,} unlabeled competition test images. Images are read in CSV order. A seed-2026 stratified 10% holdout creates disjoint training and validation index arrays. Channel means and standard deviations are computed using training images only. Validation receives normalization only and never updates optimizer or batch-normalization statistics. Hidden labels are not accessed.'
    )
    add(
        "Archive CRC checks, exact image shape, label range, unique IDs and test/sample ID order are checked before caching. The archive and split hashes are preserved in the data manifest. This competition dataset is a version of CIFAR-10; it must not be conflated with a separately evaluated standard CIFAR-10 test set [2].",
        "SmallX",
    )
    story.append(PageBreak())
    add("3. Training and Model Selection", "H1X")
    add("The model, optimizer and loss are implemented in PyTorch [4].")
    table(
        [
            ["Setting", "Value"],
            ["Optimizer", "SGD; momentum 0.9; Nesterov; weight decay 0.0005"],
            [
                "Schedule",
                "80 epochs; batch 128; initial LR 0.1; 5-epoch warm-up; cosine decay to 0.0001",
            ],
            [
                "Augmentation",
                "Reflect padding 4; independent random 32 x 32 crop; horizontal flip probability 0.5",
            ],
            ["Objective", "Cross entropy with label smoothing 0.05"],
            [
                "Selection",
                "Highest fixed-validation accuracy; ties use lower validation cross entropy",
            ],
            [
                "Initialization",
                "Kaiming-normal convolutions; no pretrained weights; seed 2026",
            ],
        ],
        [125, 365],
    )
    figure("learning_curves.png", maxh=260)
    add(
        "Figure 1. The actual epoch histories. Training accuracy includes random augmentation, and its loss includes label smoothing. Validation uses ordinary cross entropy on unaugmented images, so the two losses are not identical objectives.",
        "SmallX",
    )
    add(
        f'Validation accuracy changed from {first:.2f}% in the first epoch to a selected value of {acc:.2f}% at epoch {best}. The final epoch accuracy was {100*h[-1]["validation"]["accuracy"]:.2f}%. Checkpoint selection follows the predeclared validation rule rather than selecting the last epoch by default. The model used for inference is verified by a SHA256 hash.'
    )
    add("Optimization and recoverability", "H2X")
    add(
        "Timing covers epoch training, validation and device synchronization; checkpoint I/O is excluded. The measured hardware is an Apple M1 Pro with 16 GiB of unified memory."
    )
    add(
        "Each completed epoch writes the configuration, training/validation metrics, optimizer state and CPU/device RNG state. Checkpoints are written to a temporary file and replaced atomically. Resume checks the data hash, architecture, schedule, seed and batch size. This preserves a recoverable run, although cross-device kernels and hardware can still change floating-point results."
    )
    story.append(PageBreak())
    add("4. Validation Diagnostics", "H1X")
    figure("confusion.png", maxh=330)
    add(
        "Figure 2. Confusion matrix on the fixed validation partition, normalized within each true class. All rows are computed from reloaded-checkpoint predictions.",
        "SmallX",
    )
    classes = sorted(d["class_accuracy"].items(), key=lambda it: it[1])
    worst = classes[:3]
    bestclasses = classes[-3:]
    add("Per-class accuracy", "H2X")
    table(
        [["Class", "Accuracy", "Class", "Accuracy"]]
        + [
            [name, f"{100*v:.2f}%", name2, f"{100*v2:.2f}%"]
            for (name, v), (name2, v2) in zip(
                list(d["class_accuracy"].items())[:5],
                list(d["class_accuracy"].items())[5:],
            )
        ],
        [130, 115, 130, 115],
    )
    add(
        "The lowest validation class accuracies were "
        + ", ".join(f"{escape(k)} ({100*v:.2f}%)" for k, v in worst)
        + ". The strongest three classes were "
        + ", ".join(f"{escape(k)} ({100*v:.2f}%)" for k, v in reversed(bestclasses))
        + ". These are observations of this split and run, not claims about every possible CIFAR-10 sample."
    )
    lo, hi = d["wilson_95_interval_descriptive"]
    add(
        f"A descriptive Wilson interval for the observed proportion is [{100*lo:.2f}%, {100*hi:.2f}%]. Because the checkpoint was selected on the same validation set, this is not an independent guarantee of generalization and does not quantify all model-selection or seed uncertainty.",
        "SmallX",
    )
    story.append(PageBreak())
    add("5. Error Review and Limits", "H1X")
    figure("errors.png", maxh=390)
    add(
        "Figure 3. First 12 errors in ascending fixed validation-index order, shown with true and predicted class labels. Examples are selected mechanically rather than chosen to tell a favorable story.",
        "SmallX",
    )
    add("What this experiment supports", "H2X")
    add(
        "The saved run verifies that a residual model can be trained from scratch within the parameter budget, and that the selected checkpoint reproduces the reported validation predictions. The learning history and errors provide evidence about this one architecture and split. They do not isolate the causal contribution of width, residual connections, augmentation or smoothing, because no matched ablation study was performed."
    )
    add("Limitations and next experiments", "H2X")
    add(
        "Only one training seed and one validation split are used. Validation is also used for checkpoint selection, which can make the selected number optimistic. No ensemble or pretraining is used. A useful extension would compare narrower widths and augmentation choices with equal compute and repeat the selected configuration across seeds. The final hidden-test score, when available, is the competition evaluation rather than an interchangeable validation statistic."
    )
    story.append(PageBreak())
    add("6. Reproducibility and References", "H1X")
    add("Reproduction steps", "H2X")
    add(
        "1. Obtain the competition data.zip from Kaggle or the instructor attachment.<br/>2. Install requirements and run code/checks.py.<br/>3. Run code/prepare_data.py /path/to/data.zip --out data.<br/>4. Run code/train.py --out results/main --epochs 80.<br/>5. Run code/predict.py and code/analyze.py.<br/>6. Execute Project1.ipynb from the repository root, inspect outputs, and upload the generated CSV to Kaggle."
    )
    add(
        "Use --resume with the same settings after interruption. The prepared code package includes code, configuration, histories, summaries, figures and an executed notebook. Data archives and large checkpoints are excluded from Git; the model can be reproduced by training from scratch. The checkpoint and prediction hashes below identify this measured run. Training must occur on an allocated GPU session if using NYU HPC.",
        "BodyX",
    )
    table(
        [
            ["Evidence", "Value"],
            ["PyTorch", escape(c["torch"])],
            ["Platform", escape(c["platform"])],
            ["Data ZIP SHA256", manifest["zip_sha256"]],
            ["Checkpoint SHA256", s["checkpoint_sha256"]],
            ["Submission CSV SHA256", pred["submission_sha256"]],
        ],
        [135, 355],
    )
    if kg:
        add(
            f'Kaggle verification: {escape(str(kg.get("url","")))}<br/>Observed score: {escape(str(kg["score"]))}; submission: {escape(str(kg.get("submission_id","recorded in verification artifact")))}.',
            "SmallX",
        )
    else:
        add(
            "Kaggle score has not yet been verified. This report makes no leaderboard rank or hidden-test performance claim.",
            "SmallX",
        )
    add("References", "H2X")
    add(
        "[1] K. He, X. Zhang, S. Ren and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016. https://arxiv.org/abs/1512.03385<br/>[2] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, 2009. https://www.cs.toronto.edu/~kriz/cifar.html<br/>[3] K. Liu. pytorch-cifar. Instructor-linked background reference, consulted for context. https://github.com/kuangliu/pytorch-cifar<br/>[4] A. Paszke et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS, 2019. https://arxiv.org/abs/1912.01703",
        "SmallX",
    )
    add("Implementation and attribution", "H2X")
    add(
        "The basic-block design follows [1]. Model, data validation, training, prediction and diagnostics code were authored for this project with OpenAI Codex assistance. No downloaded model weights or external image datasets were used in the formal experiment. The course PDF defines the project constraint and report requirements.",
        "SmallX",
    )

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D8DFE8"))
        canvas.line(52, 42, 543, 42)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#5A6879"))
        canvas.drawString(52, 28, "Deep Learning Project 1 | Pengxiao Yu")
        canvas.drawRightString(543, 28, str(doc.page))

    SimpleDocTemplate(
        str(output),
        pagesize=(595.28, 841.89),
        leftMargin=52,
        rightMargin=52,
        topMargin=48,
        bottomMargin=55,
        title="Deep Learning Project 1 - Budget-Constrained CIFAR ResNet",
        author="Pengxiao Yu",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()
