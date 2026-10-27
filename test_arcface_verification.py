
from pathlib import Path

from arcface_verification import verify_faces


def main():
    print("=" * 50)
    print("       ARCFACE FACE VERIFICATION")
    print("=" * 50)

    # Get the first image path.
    image1 = input(
        "\nEnter the path to the first face image: "
    ).strip().strip('"').strip("'")

    # Get the second image path.
    image2 = input(
        "Enter the path to the second face image: "
    ).strip().strip('"').strip("'")

    try:
        # Check that both image files exist.
        if not Path(image1).is_file():
            print(f"\nError: First image not found:\n{image1}")
            return

        if not Path(image2).is_file():
            print(f"\nError: Second image not found:\n{image2}")
            return

        print("\nProcessing images...")
        print("Please wait while ArcFace compares the faces.")

        # Compare the two images.
        result = verify_faces(image1, image2)

        print("\n" + "=" * 50)
        print("          VERIFICATION RESULT")
        print("=" * 50)

        if not result["success"]:
            print(f"\nStatus: {result['message']}")
            return

        similarity = result["similarity"]
        threshold = result["threshold"]
        prediction = result["prediction"]

        print(f"\nFirst image : {image1}")
        print(f"Second image: {image2}")
        print(f"\nCosine similarity: {similarity:.4f}")
        print(f"Decision threshold: {threshold:.6f}")
        print(f"Prediction: {prediction}")

        print("\n" + "-" * 50)

        if prediction == "Same person":
            print("ArcFace considers the faces a possible match.")
        else:
            print("ArcFace considers the faces a non-match.")

        print(
            "\nNote: This is an automated prediction, "
            "not proof of identity."
        )

        print("=" * 50)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(
            f"\nUnexpected error: {error}\n"
            "Check that arcface_verification.py is saved "
            "and the required packages are installed."
        )


if __name__ == "__main__":
    main()
