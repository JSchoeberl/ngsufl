"""Turn tutorials/*.py (percent format) into executed notebooks.

    python make_notebooks.py            # all tutorials
    python make_notebooks.py poisson    # one of them
"""

import os
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

HERE = pathlib.Path(__file__).parent / "tutorials"
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"


def cells(text):
    kind, lines, out = "code", [], []

    def flush():
        body = "\n".join(lines).strip("\n")
        if body:
            if kind == "markdown":
                body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
                out.append(nbformat.v4.new_markdown_cell(body))
            else:
                out.append(nbformat.v4.new_code_cell(body))

    for line in text.split("\n"):
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return out


def build(py):
    nb = nbformat.v4.new_notebook(cells=cells(py.read_text()))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    nbformat.write(nb, py.with_suffix(".ipynb"))
    print("built", py.with_suffix(".ipynb").name)


if __name__ == "__main__":
    names = sys.argv[1:] or sorted(p.stem for p in HERE.glob("*.py"))
    for n in names:
        build(HERE / f"{n}.py")
