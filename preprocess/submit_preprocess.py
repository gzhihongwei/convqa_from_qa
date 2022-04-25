import argparse
import subprocess

from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submits `preprocess_squad.py` with SLURM."
    )
    parser.add_argument(
        "file",
        choices=["train-v2.0.json", "dev-v2.0.json"],
        type=str,
        help="The JSON to preprocess.",
    )
    parser.add_argument(
        "-p",
        "--partition",
        default="cpu",
        choices=["cpu", "cpu-long"],
        type=str,
        help="Which partition to use.",
    )
    parser.add_argument(
        "-t",
        "--time",
        default="12:00:00",
        type=str,
        help="How long to schedule the job for.",
    )
    parser.add_argument(
        "-c", "--cpus", default=2, type=int, help="How many cores of a cpu to use."
    )
    parser.add_argument(
        "-m", "--mem", default="32gb", type=str, help="How much memory to allocate."
    )
    args = parser.parse_args()

    preprocess_dir = Path(__file__).resolve().parent

    (preprocess_dir / "output" / "squad").mkdir(parents=True, exist_ok=True)

    with open(preprocess_dir / "template.sh", "r") as f:
        template = f.read()

    filled_template = template.format(
        partition=args.partition,
        time=args.time,
        cpus=args.cpus,
        memory=args.mem,
        file=args.file,
    )

    with open(preprocess_dir / "preprocess_squad.sh", "w") as f:
        f.write(filled_template)

    # Command used to submit the slurm script
    submit_command = f"sbatch {preprocess_dir / 'preprocess_squad.sh'}"

    # Submitting the job
    exit_status = subprocess.call(submit_command, shell=True)

    # Job did not submit properly
    if exit_status != 0:
        print(f"Job {submit_command} failed to submit")
