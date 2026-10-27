
from pathlib import Path
import random
import warnings

import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis
from sklearn.metrics import roc_auc_score, confusion_matrix


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("data/verification/agedb/train")
METADATA_PATH = DATASET_DIR / "metadata.csv"

MODEL_NAME = "buffalo_l"
DETECTION_SIZE = (640, 640)
DETECTION_THRESHOLD = 0.1
UPSCALE_TARGET = 512

MAX_IMAGES_PER_IDENTITY = 3
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
PREFERRED_AGE_GAP = 10

RANDOM_SEED = 42

METRICS_OUTPUT = Path("arcface_agedb_results.csv")
PAIRS_OUTPUT = Path("arcface_agedb_pair_scores.csv")
IMAGE_REPORT_OUTPUT = Path("arcface_agedb_image_report.csv")

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="insightface",
)


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def build_image_index():
    """Index actual image files, including nested AgeDB folders."""
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR.resolve()}"
        )

    image_index = {}

    for path in DATASET_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }:
            image_index.setdefault(path.name.lower(), []).append(path)

    return image_index


def resolve_image_path(file_name, image_index):
    """Resolve metadata paths without assuming a fixed subfolder."""
    name = str(file_name).replace("\\", "/")
    relative_path = Path(name)

    candidates = [
        DATASET_DIR / relative_path,
        DATASET_DIR / "images" / relative_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = image_index.get(relative_path.name.lower(), [])

    # Do not silently select an arbitrary file if filenames collide.
    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        # Prefer an exact relative suffix match when available.
        suffix_matches = [
            path for path in matches
            if path.as_posix().lower().endswith(
                "/" + name.lower().lstrip("/")
            )
        ]

        if len(suffix_matches) == 1:
            return suffix_matches[0].resolve()

    return None


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH.resolve()}"
        )

    df = pd.read_csv(METADATA_PATH)

    required = {"file_name", "identity", "age"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing)}"
        )

    image_index = build_image_index()

    df = df.copy()
    df["identity"] = df["identity"].astype("string").str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

    df = df.dropna(subset=["identity", "age"])
    df = df[df["identity"] != ""]

    resolved_paths = []
    for file_name in df["file_name"]:
        resolved_paths.append(
            resolve_image_path(file_name, image_index)
        )

    df["path"] = resolved_paths

    missing_count = int(df["path"].isna().sum())
    if missing_count:
        print("Metadata rows with unresolved image paths:", missing_count)

    df = df.dropna(subset=["path"])
    df = df.drop_duplicates(subset=["path"])
    df = df.reset_index(drop=True)

    print("Valid image records:", len(df))
    print("Unique identities:", df["identity"].nunique())

    if df.empty:
        raise RuntimeError(
            "No valid images found. Check the metadata and dataset paths."
        )

    return df


# ============================================================
# IDENTITY-DISJOINT SPLIT
# ============================================================

def split_identities(df):
    counts = df.groupby("identity").size()
    eligible_ids = counts[counts >= 2].index.to_numpy().copy()

    if len(eligible_ids) < 10:
        raise RuntimeError(
            "Not enough identities with at least two images."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(eligible_ids)

    total = len(eligible_ids)
    n_test = max(1, round(total * TEST_FRACTION))
    n_val = max(1, round(total * VALIDATION_FRACTION))

    if n_test + n_val >= total:
        raise RuntimeError("Not enough identities for all three splits.")

    test_ids = set(eligible_ids[:n_test])
    val_ids = set(eligible_ids[n_test:n_test + n_val])
    dev_ids = set(eligible_ids[n_test + n_val:])

    assert not dev_ids & val_ids
    assert not dev_ids & test_ids
    assert not val_ids & test_ids

    print("\nIDENTITY SPLIT")
    print("Development identities:", len(dev_ids))
    print("Validation identities:", len(val_ids))
    print("Test identities:", len(test_ids))

    return dev_ids, val_ids, test_ids


# ============================================================
# SELECT AGE-DIVERSE IMAGES
# ============================================================

def select_images(df, identities):
    selected = []

    subset = df[df["identity"].isin(identities)]

    for identity, group in subset.groupby("identity"):
        group = group.sort_values(
            ["age", "path"]
        ).drop_duplicates(subset=["path"])

        if len(group) < 2:
            continue

        if len(group) <= MAX_IMAGES_PER_IDENTITY:
            chosen = group
        else:
            positions = np.linspace(
                0,
                len(group) - 1,
                MAX_IMAGES_PER_IDENTITY,
            )
            positions = np.round(positions).astype(int)
            chosen = group.iloc[positions]

        selected.extend(chosen.to_dict("records"))

    result = pd.DataFrame(selected)

    if result.empty:
        raise RuntimeError("No images selected for this split.")

    return result.reset_index(drop=True)


# ============================================================
# READ AND PREPARE IMAGES
# ============================================================

def read_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)

    if data.size == 0:
        return None

    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def prepare_image(image):
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side < UPSCALE_TARGET:
        scale = UPSCALE_TARGET / longest_side
        image = cv2.resize(
            image,
            (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            ),
            interpolation=cv2.INTER_CUBIC,
        )

    return image


# ============================================================
# EXTRACT EACH IMAGE EMBEDDING ONCE
# ============================================================

def extract_embeddings(df):
    print("\nLoading ArcFace model...")

    face_app = FaceAnalysis(
        name=MODEL_NAME,
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(
        ctx_id=-1,
        det_size=DETECTION_SIZE,
        det_thresh=DETECTION_THRESHOLD,
    )

    print("ArcFace loaded.")
    print("Detection size:", DETECTION_SIZE)
    print("Detection threshold:", DETECTION_THRESHOLD)

    embeddings = {}
    report = []

    stats = {
        "selected": len(df),
        "upscaled": 0,
        "read_failures": 0,
        "no_face": 0,
        "processing_failures": 0,
        "multiple_faces": 0,
        "embeddings_extracted": 0,
    }

    for position, row in enumerate(df.itertuples(index=False), start=1):
        path = Path(row.path)
        identity = str(row.identity)
        age = float(row.age)
        original_height = None
        original_width = None
        face_count = 0
        selected_score = None
        status = "ok"

        try:
            image = read_image(path)

            if image is None:
                stats["read_failures"] += 1
                status = "read_failure"
            else:
                original_height, original_width = image.shape[:2]

                if max(original_height, original_width) < UPSCALE_TARGET:
                    stats["upscaled"] += 1

                image = prepare_image(image)
                faces = face_app.get(image)
                face_count = len(faces)

                if not faces:
                    stats["no_face"] += 1
                    status = "no_face"
                else:
                    if len(faces) > 1:
                        stats["multiple_faces"] += 1

                    # Heuristic: choose the largest detected face.
                    # Multiple-face images should be manually checked.
                    face = max(
                        faces,
                        key=lambda item: max(
                            0.0, float(item.bbox[2] - item.bbox[0])
                        ) * max(
                            0.0, float(item.bbox[3] - item.bbox[1])
                        ),
                    )

                    embedding = np.asarray(
                        face.normed_embedding,
                        dtype=np.float32,
                    )

                    norm = float(np.linalg.norm(embedding))

                    if (
                        embedding.ndim != 1
                        or not np.isfinite(embedding).all()
                        or not np.isfinite(norm)
                        or norm <= 0
                    ):
                        stats["processing_failures"] += 1
                        status = "invalid_embedding"
                    else:
                        # Key embeddings by the unique resolved image path.
                        key = str(path.resolve())
                        embeddings[key] = embedding / norm
                        selected_score = float(face.det_score)
                        stats["embeddings_extracted"] += 1

        except Exception as exc:
            stats["processing_failures"] += 1
            status = "processing_failure"
            print(
                f"Error on {path.name}: {str(exc)[:150]}"
            )

        report.append({
            "path": str(path.resolve()),
            "identity": identity,
            "age": age,
            "width": original_width,
            "height": original_height,
            "faces_detected": face_count,
            "selected_face_score": selected_score,
            "status": status,
        })

        if position % 50 == 0 or position == len(df):
            print(f"Processed {position}/{len(df)} images...")

    report_df = pd.DataFrame(report)
    report_df.to_csv(IMAGE_REPORT_OUTPUT, index=False)

    print("\nIMAGE DIAGNOSTICS")
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    success_rate = (
        100.0 * len(embeddings) / len(df)
        if len(df) else 0.0
    )
    print(f"Embedding success rate: {success_rate:.2f}%")
    print("Image report saved to:", IMAGE_REPORT_OUTPUT.resolve())

    return embeddings, report_df


# ============================================================
# CREATE GENUINE AND IMPOSTOR PAIRS
# ============================================================

def create_pairs(df, embeddings, split_name):
    usable = df[
        df["path"].apply(lambda path: str(Path(path).resolve()) in embeddings)
    ].copy()

    if usable.empty:
        raise RuntimeError(f"{split_name}: no usable image embeddings.")

    usable["path_key"] = usable["path"].apply(
        lambda path: str(Path(path).resolve())
    )

    genuine_preferred = []
    genuine_fallback = []

    # Same-identity comparisons.
    for identity, group in usable.groupby("identity", sort=True):
        records = group.sort_values(["age", "path_key"]).to_dict("records")

        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                a, b = records[i], records[j]
                gap = abs(float(a["age"]) - float(b["age"]))

                pair = (a, b, gap)

                if gap >= PREFERRED_AGE_GAP:
                    genuine_preferred.append(pair)
                else:
                    genuine_fallback.append(pair)

    if not genuine_preferred and not genuine_fallback:
        raise RuntimeError(f"{split_name}: no genuine pairs available.")

    # Use age-separated pairs when possible. Keep all genuine pairs
    # when there are only a few selected images per identity.
    genuine_pairs = genuine_preferred + genuine_fallback

    identities = sorted(usable["identity"].unique().tolist())

    if len(identities) < 2:
        raise RuntimeError(
            f"{split_name}: need at least two identities for impostor pairs."
        )

    groups = {
        identity: group.to_dict("records")
        for identity, group in usable.groupby("identity", sort=True)
    }

    # Sample distinct impostor image pairs reproducibly.
    rng = random.Random(f"{RANDOM_SEED}:{split_name}")
    target = len(genuine_pairs)
    impostor_by_key = {}

    max_attempts = max(10000, target * 100)
    attempts = 0

    while len(impostor_by_key) < target and attempts < max_attempts:
        attempts += 1

        id_a, id_b = rng.sample(identities, 2)
        a = rng.choice(groups[id_a])
        b = rng.choice(groups[id_b])

        key_a = str(Path(a["path"]).resolve())
        key_b = str(Path(b["path"]).resolve())

        if key_a == key_b:
            continue

        pair_key = tuple(sorted((key_a, key_b)))

        if pair_key in impostor_by_key:
            continue

        gap = abs(float(a["age"]) - float(b["age"]))
        impostor_by_key[pair_key] = (a, b, gap)

    # If random sampling did not find enough unique pairs, use a
    # deterministic fallback over cross-identity combinations.
    if len(impostor_by_key) < target:
        for i, id_a in enumerate(identities):
            for id_b in identities[i + 1:]:
                for a in groups[id_a]:
                    for b in groups[id_b]:
                        key_a = str(Path(a["path"]).resolve())
                        key_b = str(Path(b["path"]).resolve())
                        pair_key = tuple(sorted((key_a, key_b)))

                        if pair_key not in impostor_by_key:
                            gap = abs(float(a["age"]) - float(b["age"]))
                            impostor_by_key[pair_key] = (a, b, gap)

                        if len(impostor_by_key) >= target:
                            break
                    if len(impostor_by_key) >= target:
                        break
                if len(impostor_by_key) >= target:
                    break
            if len(impostor_by_key) >= target:
                break

    impostor_pairs = list(impostor_by_key.values())[:target]

    if not impostor_pairs:
        raise RuntimeError(f"{split_name}: no impostor pairs available.")

    # Score all pairs. Store full pair-level results for auditability.
    pair_rows = []

    for label, pairs in ((1, genuine_pairs), (0, impostor_pairs)):
        for a, b, age_gap in pairs:
            path_a = str(Path(a["path"]).resolve())
            path_b = str(Path(b["path"]).resolve())

            score = float(np.dot(embeddings[path_a], embeddings[path_b]))

            pair_rows.append({
                "split": split_name,
                "label": label,
                "pair_type": "genuine" if label == 1 else "impostor",
                "path_a": path_a,
                "path_b": path_b,
                "identity_a": str(a["identity"]),
                "identity_b": str(b["identity"]),
                "age_a": float(a["age"]),
                "age_b": float(b["age"]),
                "age_gap": float(age_gap),
                "similarity": score,
            })

    pairs_df = pd.DataFrame(pair_rows)

    print(f"\n{split_name} PAIRS")
    print("Genuine pairs:", int((pairs_df["label"] == 1).sum()))
    print("Impostor pairs:", int((pairs_df["label"] == 0).sum()))

    return pairs_df


# ============================================================
# SELECT THRESHOLD USING VALIDATION ONLY
# ============================================================

def find_best_threshold(scores, labels):
    if len(scores) == 0:
        raise RuntimeError("No validation scores available.")

    candidates = np.unique(scores)

    if len(candidates) > 1:
        midpoints = (candidates[:-1] + candidates[1:]) / 2
        candidates = np.concatenate([
            [candidates[0] - 1e-6],
            midpoints,
            [candidates[-1] + 1e-6],
        ])

    best_threshold = None
    best_bacc = -1.0

    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)

        genuine = labels == 1
        impostor = labels == 0

        if not genuine.any() or not impostor.any():
            raise RuntimeError(
                "Threshold selection needs both pair classes."
            )

        tar = float(np.mean(predicted[genuine] == 1))
        trr = float(np.mean(predicted[impostor] == 0))
        bacc = (tar + trr) / 2

        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = float(threshold)

    return best_threshold, best_bacc


# ============================================================
# METRICS
# ============================================================

def report_metrics(title, pairs_df, threshold):
    scores = pairs_df["similarity"].to_numpy(dtype=float)
    labels = pairs_df["label"].to_numpy(dtype=int)

    if len(scores) == 0 or len(np.unique(labels)) != 2:
        raise RuntimeError(
            f"{title} must contain genuine and impostor pairs."
        )

    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])

    tn, fp, fn, tp = matrix.ravel()

    far = fp / (fp + tn) if fp + tn else 0.0
    frr = fn / (fn + tp) if fn + tp else 0.0
    tar = tp / (tp + fn) if tp + fn else 0.0
    trr = tn / (tn + fp) if tn + fp else 0.0

    accuracy = float(np.mean(predictions == labels))
    balanced_accuracy = (tar + trr) / 2
    auc = float(roc_auc_score(labels, scores))

    genuine = pairs_df[pairs_df["label"] == 1]
    impostor = pairs_df[pairs_df["label"] == 0]

    print(f"\n{title}")
    print("=" * len(title))
    print("Total pairs:", len(pairs_df))
    print("Genuine pairs:", len(genuine))
    print("Impostor pairs:", len(impostor))
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced accuracy: {balanced_accuracy:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"FAR: {far:.4f}")
    print(f"FRR: {frr:.4f}")
    print(f"TAR: {tar:.4f}")
    print(f"TRR: {trr:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(matrix)

    print(
        "Mean genuine age gap:",
        f"{genuine['age_gap'].mean():.2f} years",
    )
    print(
        "Mean impostor age gap:",
        f"{impostor['age_gap'].mean():.2f} years",
    )

    # Additional descriptive analysis by genuine-pair age gap.
    print("\nGenuine-pair age-gap analysis:")
    bins = [0, 5, 10, 20, 30, 50, np.inf]
    labels_bins = ["0-4", "5-9", "10-19", "20-29", "30-49", "50+"]

    genuine_scored = genuine.copy()
    genuine_scored["accepted"] = (
        genuine_scored["similarity"] >= threshold
    )

    genuine_scored["age_gap_bin"] = pd.cut(
        genuine_scored["age_gap"],
        bins=bins,
        labels=labels_bins,
        right=False,
        include_lowest=True,
    )

    for age_bin, group in genuine_scored.groupby(
        "age_gap_bin", observed=False
    ):
        if len(group):
            rate = float(group["accepted"].mean())
            print(
                f"  {age_bin} years: "
                f"{len(group)} pairs, TAR={rate:.4f}"
            )

    return {
        "split": title,
        "pairs": len(pairs_df),
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": auc,
        "far": far,
        "frr": frr,
        "tar": tar,
        "trr": trr,
        "mean_genuine_age_gap": float(genuine["age_gap"].mean()),
        "mean_impostor_age_gap": float(impostor["age_gap"].mean()),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_metadata()

    counts = df.groupby("identity").size()
    eligible_ids = set(counts[counts >= 2].index)

    df = df[df["identity"].isin(eligible_ids)].reset_index(drop=True)

    dev_ids, val_ids, test_ids = split_identities(df)

    # Keep development identities separate; this evaluation uses
    # validation for threshold selection and test for final metrics.
    val_df = select_images(df, val_ids)
    test_df = select_images(df, test_ids)

    combined = pd.concat(
        [val_df, test_df],
        ignore_index=True,
    ).drop_duplicates(subset=["path"]).reset_index(drop=True)

    print("\nSELECTED IMAGES")
    print("Validation images:", len(val_df))
    print("Test images:", len(test_df))
    print("Unique images to process:", len(combined))

    # One embedding extraction pass for all validation and test images.
    embeddings, image_report = extract_embeddings(combined)

    val_pairs = create_pairs(val_df, embeddings, "validation")
    test_pairs = create_pairs(test_df, embeddings, "test")

    val_scores = val_pairs["similarity"].to_numpy(dtype=float)
    val_labels = val_pairs["label"].to_numpy(dtype=int)

    threshold, val_bacc = find_best_threshold(
        val_scores,
        val_labels,
    )

    print("\nTHRESHOLD SELECTED ON VALIDATION DATA")
    print(f"Threshold: {threshold:.4f}")
    print(f"Validation balanced accuracy: {val_bacc:.4f}")

    validation_results = report_metrics(
        "VALIDATION RESULTS",
        val_pairs,
        threshold,
    )

    test_results = report_metrics(
        "HELD-OUT TEST RESULTS",
        test_pairs,
        threshold,
    )

    # Save metrics and all individual pair scores.
    pd.DataFrame([
        validation_results,
        test_results,
    ]).to_csv(METRICS_OUTPUT, index=False)

    all_pairs = pd.concat(
        [val_pairs, test_pairs],
        ignore_index=True,
    )
    all_pairs.to_csv(PAIRS_OUTPUT, index=False)

    print("\nSaved metrics:", METRICS_OUTPUT.resolve())
    print("Saved pair scores:", PAIRS_OUTPUT.resolve())
    print("Saved image report:", IMAGE_REPORT_OUTPUT.resolve())

    print(
        "\nCaution: results are preliminary until face boxes and "
        "multiple-face selections are visually validated. Pair scores "
        "share images, so individual pairs are not statistically independent."
    )


if __name__ == "__main__":
    main()
