"""Write a private Kaggle kernel that runs ``run_cuda.ipynb`` for one commit and some Reynolds numbers.

    python kaggle/prepare_kernel.py --ref <40-character SHA> --reynolds 100 --run-full --out /tmp/kernel-re100
    kaggle kernels push -p /tmp/kernel-re100

The tracked notebook is not modified: the kernel directory gets a copy in which only the
settings of the first code cell are replaced, plus ``kernel-metadata.json`` (private, GPU and
internet on). Outputs are downloaded afterwards with ``kaggle kernels output <id> -p <dir>``.
"""

import argparse
import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent / "run_cuda.ipynb"


def parametrise(cell: str, ref: str, reynolds: list[int], run_full: bool, budget_hours: float) -> str:
    settings = {
        "REF": f'"{ref}"',
        "RUN_FULL": str(run_full),
        "REYNOLDS": str(sorted(reynolds)),
        "SESSION_BUDGET_HOURS": str(budget_hours),
    }
    for name, value in settings.items():
        cell, n = re.subn(rf"^{name} = [^#\n]*?(\s*#|$)", rf"{name} = {value}\1", cell, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"setting {name} not found in the first code cell of {NOTEBOOK}")
    return cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="full commit SHA, pushed to GitHub")
    parser.add_argument("--reynolds", type=int, nargs="+", required=True)
    parser.add_argument("--run-full", action="store_true", help="set RUN_FULL = True (step B)")
    parser.add_argument("--budget-hours", type=float, default=11.0)
    parser.add_argument("--user", default="adebanjiadelowo")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.ref):
        raise SystemExit("--ref must be a full 40-character commit SHA")

    nb = json.loads(NOTEBOOK.read_text())
    first = next(c for c in nb["cells"] if c["cell_type"] == "code")
    first["source"] = parametrise("".join(first["source"]), args.ref, args.reynolds, args.run_full, args.budget_hours).splitlines(keepends=True)
    tag = "re" + "-".join(str(r) for r in sorted(args.reynolds))
    slug = f"physicsnemo-ldc-pinns-{'full' if args.run_full else 'checks'}-{tag}-{args.ref[:7]}"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_cuda.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
    metadata = {
        "id": f"{args.user}/{slug}",
        "title": slug,
        "code_file": "run_cuda.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (args.out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metadata["id"])


if __name__ == "__main__":
    main()
