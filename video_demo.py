"""CLI wrapper around the shared demo engine.

Usage:
    python video_demo.py --video input.mp4 --out outputs/demo_annotated.mp4
    python video_demo.py --video input.mp4 --ocr-every 1 --show
"""

import argparse

import demo_engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Input video file")
    ap.add_argument("--out", default="outputs/demo_annotated.mp4", help="Output video file")
    ap.add_argument("--ocr-every", type=int, default=1, help="Run plate OCR every N frames")
    ap.add_argument("--yolo-every", type=int, default=1, help="Run YOLO every N frames")
    ap.add_argument("--max-width", type=int, default=1280, help="Downscale frames to this width")
    ap.add_argument("--show", action="store_true",
                    help="Show a live detection window while processing (q/ESC to quit)")
    ap.add_argument("--camera", type=int, metavar="ID", default=None,
                    help="Real-time webcam mode instead of a file (ID=0 is default webcam)")
    args = ap.parse_args()

    if args.camera is not None and not args.video:
        def on_status(msg):
            print(msg)

        print("Starting real-time detection. Press q or ESC in the window to quit.")
        demo_engine.process_camera(
            camera_index=args.camera, ocr_every=args.ocr_every, yolo_every=args.yolo_every,
            max_width=args.max_width, on_status=on_status)
        return

    if not args.video:
        ap.error("provide --video <file> or --camera <id>")
        return

    def on_status(msg):
        print(msg)

    def on_progress(done, total):
        print(f"  processed {done}/{total} frames")

    out, plates, summary = demo_engine.process_video(
        args.video, args.out, ocr_every=args.ocr_every, yolo_every=args.yolo_every,
        max_width=args.max_width, on_status=on_status,
        on_progress=on_progress, show=args.show)

    print(f"\nAnnotated video saved to: {out}")
    print(f"Plates read: {plates}")
    print(f"Summary -> vehicles: {summary['vehicles']}, "
          f"plates located: {summary['plates_located']}, "
          f"plates read: {summary['plates_read']}")


if __name__ == "__main__":
    main()
