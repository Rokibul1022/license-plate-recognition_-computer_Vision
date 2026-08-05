"""
One-time download of a ready-to-use, pretrained YOLOv11 license plate detector.
No training or fine-tuning required.

Model: morsetechlab/yolov11-license-plate-detection (Hugging Face)
- Fine-tuned on 10,125 images from Roboflow's license-plate-recognition-rxg4e dataset
- Choose model size based on your RTX 3060: 's' or 'm' is a good speed/accuracy balance,
  'n' is fastest, 'x' is most accurate but slowest.

Usage:
    python download_plate_model.py --size s
    # downloads to ./models/plate_detector_yolov11s.pt
"""
import argparse
import os
from huggingface_hub import hf_hub_download

REPO_ID = "morsetechlab/yolov11-license-plate-detection"

# filenames as hosted in the repo (pytorch weights)
SIZE_TO_FILENAME = {
    "n": "license-plate-finetune-v1n.pt",
    "s": "license-plate-finetune-v1s.pt",
    "m": "license-plate-finetune-v1m.pt",
    "l": "license-plate-finetune-v1l.pt",
    "x": "license-plate-finetune-v1x.pt",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SIZE_TO_FILENAME.keys()), default="s",
                         help="Model size: n(ano)/s(mall)/m(edium)/l(arge)/x(large). "
                              "s or m recommended for RTX 3060.")
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    filename = SIZE_TO_FILENAME[args.size]

    print(f"Downloading {filename} from {REPO_ID} ...")
    local_path = hf_hub_download(repo_id=REPO_ID, filename=filename,
                                  local_dir=args.output_dir)
    print(f"Done. Weights saved to: {local_path}")
    print(f"\nRun the pipeline with:\n"
          f"  python pipeline.py --video your_4k_video.mp4 "
          f"--plate-model {local_path} --device cuda:0")


if __name__ == "__main__":
    main()
