from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_do_not_expose_local_machine_paths():
    forbidden_markers = (
        "/Users/",
        "C:\\Users\\",
    )

    for notebook in (REPOSITORY_ROOT / "notebooks").glob("*.ipynb"):
        contents = notebook.read_text(encoding="utf-8")

        for marker in forbidden_markers:
            assert marker not in contents, f"{notebook.name} contains {marker}"
