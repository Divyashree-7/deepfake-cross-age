
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# Use the threshold selected by the existing validation evaluation.
SIMILARITY_THRESHOLD = 0.160528

# Initialize ArcFace once when this module is imported.
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)
face_app.prepare(
    ctx_id=-1,
    det_size=(640, 640),
    det_thresh=0.1
)


def load_image(image_path):
    """Load an image safely, including paths with Unicode characters."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image



def extract_face_embedding(image_path):
    """
    Return the embedding for the largest detected face.
    Print diagnostic information if face detection fails.
    """
    image = load_image(image_path)
    faces = face_app.get(image)

    print(f"\nImage: {image_path}")
    print(f"Faces detected: {len(faces)}")

    if len(faces) == 0:
        print("WARNING: No face detected in this image.")
        return None

    # Select the largest detected face.
    face = max(
        faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )
    )

    print("Selected face bounding box:", face.bbox)

    embedding = face.normed_embedding
    embedding = np.asarray(embedding, dtype=np.float32)

    return embedding




def verify_faces(image1_path, image2_path):
    """Compare two images and report the similarity score."""
    embedding1 = extract_face_embedding(image1_path)
    embedding2 = extract_face_embedding(image2_path)

    if embedding1 is None or embedding2 is None:
        return {
            "success": False,
            "message": "No face detected in one or both images.",
            "similarity": None,
            "threshold": SIMILARITY_THRESHOLD,
            "prediction": None
        }

    similarity = float(np.dot(embedding1, embedding2))

    print(f"Cosine similarity: {similarity:.6f}")
    print(f"Current threshold: {SIMILARITY_THRESHOLD:.6f}")

    prediction = (
        "Same person"
        if similarity >= SIMILARITY_THRESHOLD
        else "Different people"
    )

    print(f"Prediction: {prediction}")

    return {
        "success": True,
        "message": "Comparison completed.",
        "similarity": similarity,
        "threshold": SIMILARITY_THRESHOLD,
        "prediction": prediction
    }


   