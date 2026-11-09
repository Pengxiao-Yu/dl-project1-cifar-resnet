"""Execute the notebook in a real local kernel, with a temporary kernel spec.
Does not register or modify a persistent user Jupyter kernel.
"""

import json
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / "results/main/summary.json").read_text())
    assert summary["complete"] and summary["epochs_completed"] == 80
    notebook = nbformat.read(ROOT / "Project1.ipynb", as_version=4)
    with tempfile.TemporaryDirectory(prefix="project1-kernel-") as temp:
        kernel_dir = Path(temp) / "project1"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Project1 isolated kernel",
                    "language": "python",
                }
            )
        )
        manager = AsyncKernelManager(
            kernel_name="project1",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp]),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=900,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute(cleanup_kc=True)
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == "error" for c in cells for o in c.outputs)
    nbformat.write(notebook, ROOT / "Project1.ipynb")
    print("Executed and verified", len(cells), "code cells without errors.")


if __name__ == "__main__":
    main()
