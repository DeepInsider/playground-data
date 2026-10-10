"""Execute maintained article copies in the selected uv environment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    "base": [
        "basic",
        "deeplearning1g1t_lesson08",
        "deeplearning1g2t_lesson01",
        "deeplearningdat",
    ],
    "tensorflow": [
        "deeplearning1g1t_lesson07",
        "deeplearningpyt",
        "tf2_keras_neuralnetwork",
        "tf2_keras_regression",
        "tf2_keras_howtowrite",
    ],
    "pytorch": ["pytorch_neuralnetwork"],
}


def prepare(source):
    nb = nbformat.read(source, as_version=4)
    nb.metadata.kernelspec = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.metadata["validation_source"] = str(source.relative_to(ROOT)).replace("\\", "/")
    nb.metadata["validation_source_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    for i, cell in enumerate(nb.cells):
        cell.metadata["original_cell_index"] = i
        if cell.cell_type != "code":
            continue
        cell.outputs = []
        cell.execution_count = None
        lines = []
        for line in cell.source.splitlines():
            if line.lstrip().startswith(("!", "%tensorflow_version")):
                line = (
                    "# Environment/shell command handled by the uv validation environment: "
                    + line
                )
            lines.append(line)
        cell.source = "\n".join(lines)
        if "from google.colab import files" in cell.source:
            cell.source = "from IPython.display import FileLink\ndisplay(FileLink('training.log'))"
    patches_path = ROOT / "notebooks/patches.json"
    patches = (
        json.loads(patches_path.read_text(encoding="utf-8"))
        if patches_path.exists()
        else {}
    )
    for cell in nb.cells:
        patch = patches.get(source.stem, {}).get(
            str(cell.metadata.get("original_cell_index"))
        )
        if patch:
            cell.source = patch["source"]
            cell.metadata["compatibility_reason"] = patch["reason"]
            if "expected_error" in patch:
                cell.metadata["expected_error"] = patch["expected_error"]
    setup = nbformat.v4.new_code_cell(
        "import sys, os, random\nfrom pathlib import Path\nimport importlib.util\nroot = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / 'plygdata').is_dir())\nos.environ['MPLCONFIGDIR'] = str(root / '.artifacts/notebooks/matplotlib')\nos.environ['KERAS_HOME'] = str(root / '.artifacts/notebooks/keras')\ngraphviz = root / '.artifacts/graphviz/Graphviz-16.1.0-win64/bin'\nif graphviz.exists():\n    os.environ['PATH'] = str(graphviz) + os.pathsep + os.environ['PATH']\nimport numpy as np\nimport plygdata as pg\nassert pg.__version__ == '1.2.0'\nget_ipython().run_line_magic('matplotlib', 'inline')\nrandom.seed(42)\nnp.random.seed(42)\nif importlib.util.find_spec('tensorflow'):\n    import tensorflow as tf\n    tf.random.set_seed(42)\nif importlib.util.find_spec('torch'):\n    import torch\n    torch.manual_seed(42)\nprint('Python:', sys.version.split()[0], 'playground-data:', pg.__version__)\nprint('Package:', pg.__file__)\n"
    )
    nb.cells.insert(
        0,
        nbformat.v4.new_markdown_cell(
            "# Maintained execution copy\n\nThe article original is preserved. Use the matching locked uv environment.\nEnvironment-install cells are disabled here; all learning parameters are retained."
        ),
    )
    nb.cells.insert(1, setup)
    nb.nbformat_minor = 5
    for i, cell in enumerate(nb.cells):
        cell.id = f"cell-{i:04d}"
    nbformat.validate(nb)
    return nb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".artifacts/notebooks/matplotlib")
    os.environ["KERAS_HOME"] = str(ROOT / ".artifacts/notebooks/keras")
    graphviz = ROOT / ".artifacts/graphviz/Graphviz-16.1.0-win64/bin"
    if graphviz.exists():
        os.environ["PATH"] = str(graphviz) + os.pathsep + os.environ["PATH"]
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
    os.environ["TF_NUM_INTEROP_THREADS"] = "2"
    results = []
    for name in args.only or GROUPS[args.group]:
        if name not in GROUPS[args.group]:
            parser.error(f"{name} is not in the {args.group} group")
        source = next(f for f in ROOT.glob("docs/**/*.ipynb") if f.stem == name)
        dest = ROOT / "notebooks/validated" / (name + ".ipynb")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if args.prepare or not dest.exists():
            nbformat.write(prepare(source), dest)
        nb = nbformat.read(dest, as_version=4)
        nbformat.validate(nb)
        if hashlib.sha256(source.read_bytes()).hexdigest() != nb.metadata.get(
            "validation_source_sha256"
        ):
            raise ValueError(
                f"Article source changed: regenerate {name} with --prepare"
            )
        work = ROOT / ".artifacts/notebooks/work" / name
        work.mkdir(parents=True, exist_ok=True)
        km = KernelManager(kernel_name="python3")
        km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        client = NotebookClient(
            nb,
            km=km,
            timeout=600,
            allow_errors=True,
            resources={"metadata": {"path": str(work)}},
        )
        print("RUN", name, flush=True)
        started = time.monotonic()
        fatal = None
        try:
            client.execute(cwd=str(work))
        except Exception as exc:  # noqa: BLE001 - record kernel failures, then fail the validation run
            fatal = type(exc).__name__ + ": " + str(exc)
        finally:
            if km.has_kernel:
                km.shutdown_kernel(now=True)
        out = ROOT / ".artifacts/notebooks/executed" / (name + ".ipynb")
        out.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, out)
        errors = []
        expected_errors = []
        for cell in nb.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "error":
                    target = (
                        expected_errors
                        if cell.metadata.get("expected_error") == output.ename
                        else errors
                    )
                    target.append(
                        {
                            "cell": cell.metadata.get("original_cell_index"),
                            "type": output.ename,
                            "message": output.evalue,
                        }
                    )
        presentation = nbformat.reads(nbformat.writes(nb), as_version=4)
        for cell in presentation.cells:
            for output in cell.get("outputs", []):
                if output.output_type == "stream" and len(output.text) > 5000:
                    output.text = (
                        "[Long training log abbreviated in HTML; full output is in the executed notebook.]\n"
                        + output.text[-4000:]
                    )
        for cell in nb.cells:
            expected = cell.metadata.get("expected_error")
            if expected and not any(
                o.output_type == "error" and o.ename == expected
                for o in cell.get("outputs", [])
            ):
                errors.append(
                    {
                        "cell": cell.metadata.get("original_cell_index"),
                        "type": "ExpectedErrorNotObserved",
                        "message": expected,
                    }
                )
        html, _ = HTMLExporter().from_notebook_node(presentation)
        out.with_suffix(".html").write_text(html, encoding="utf-8")
        versions = {}
        for package in [
            "playground-data",
            "numpy",
            "matplotlib",
            "pandas",
            "tensorflow",
            "keras",
            "torch",
            "torchvision",
            "nbclient",
            "ipykernel",
            "jupyterlab",
        ]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        result = {
            "notebook": name,
            "group": args.group,
            "python": sys.version.split()[0],
            "versions": versions,
            "seconds": round(time.monotonic() - started, 1),
            "errors": errors,
            "expected_errors": expected_errors,
            "fatal": fatal,
            "executed_code_cells": sum(
                c.cell_type == "code" and c.execution_count is not None
                for c in nb.cells
            ),
        }
        results.append(result)
        report = ROOT / ".artifacts/notebooks" / (args.group + "-results.json")
        report.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=True), flush=True)
    if any(r["fatal"] or r["errors"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
