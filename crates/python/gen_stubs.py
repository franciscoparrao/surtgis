#!/usr/bin/env python3
"""Regenerate surtgis.pyi from the #[pyfunction] signatures in src/lib.rs
and src/cloud.rs.

Run from crates/python/ after adding/changing a #[pyfunction] binding:

    python3 gen_stubs.py > surtgis.pyi

Parses Rust source with regexes tuned to this crate's existing
`#[pyfunction]` / `#[pyo3(signature = (...))]` conventions — not a general
Rust parser. If a new binding uses an unhandled type shape, the affected
parameter/return falls back to `typing.Any` rather than failing; check
the diff after regenerating.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

TYPE_MAP = {
    "f64": "float",
    "f32": "float",
    "usize": "int",
    "u64": "int",
    "u32": "int",
    "u16": "int",
    "u8": "int",
    "i64": "int",
    "i32": "int",
    "bool": "bool",
    "&str": "str",
    "String": "str",
}

NUMPY_DTYPE = {
    "f64": "float64",
    "f32": "float32",
    "u8": "uint8",
    "u16": "uint16",
    "u32": "uint32",
    "u64": "uint64",
    "i32": "int32",
    "i64": "int64",
}


def rust_type_to_py(t: str) -> str:
    t = t.strip()
    m = re.match(r"PyReadonlyArray[123]<'py,\s*(\w+)>", t)
    if m:
        return f"npt.NDArray[np.{NUMPY_DTYPE.get(m.group(1), m.group(1))}]"
    m = re.match(r"Option<PyReadonlyArray[123]<'py,\s*(\w+)>>", t)
    if m:
        return f"npt.NDArray[np.{NUMPY_DTYPE.get(m.group(1), m.group(1))}] | None"
    m = re.match(r"Vec<PyReadonlyArray[123]<'py,\s*(\w+)>>", t)
    if m:
        return f"list[npt.NDArray[np.{NUMPY_DTYPE.get(m.group(1), m.group(1))}]]"
    if t == "&Bound<'py, PyAny>":
        return "typing.Callable[..., typing.Any]"
    m = re.match(r"Option<(.+)>$", t)
    if m:
        return f"{rust_type_to_py(m.group(1))} | None"
    m = re.match(r"Vec<(.+)>$", t)
    if m:
        return f"list[{rust_type_to_py(m.group(1))}]"
    if t.startswith("(") and t.endswith(")"):
        parts = split_top_level(t[1:-1])
        if parts:
            return "tuple[" + ", ".join(rust_type_to_py(p) for p in parts) + "]"
    if t in TYPE_MAP:
        return TYPE_MAP[t]
    return "typing.Any"


def ret_type_to_py(t: str) -> str:
    t = t.strip()
    m = re.match(r"PyResult<(.*)>$", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    if t == "()":
        return "None"
    if t.startswith("("):
        inner = t[1:-1]
        parts = split_top_level(inner)
        return "tuple[" + ", ".join(ret_type_to_py(p) for p in parts) + "]"
    m = re.match(r"Bound<'py,\s*PyArray[123]<(\w+)>>", t)
    if m:
        return f"npt.NDArray[np.{NUMPY_DTYPE.get(m.group(1), m.group(1))}]"
    m = re.match(r"Bound<'py,\s*(?:pyo3::types::)?PyList>", t)
    if m:
        return "list[dict[str, typing.Any]]"
    m = re.match(r"Bound<'py,\s*(?:pyo3::types::)?PyDict>", t)
    if m:
        return "dict[str, typing.Any]"
    if t in TYPE_MAP:
        return TYPE_MAP[t]
    return "typing.Any"


def split_top_level(s: str):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur)
    return parts


def parse_file(path: Path):
    src = path.read_text()
    funcs = []
    pattern = re.compile(
        r"#\[pyfunction\]\s*"
        r"(?:#\[pyo3\(signature\s*=\s*\((?P<sig>.*?)\)\)\]\s*)?"
        r"(?:#\[allow\([^\]]*\)\]\s*)?"
        r"fn\s+(?P<name>\w+)\s*(?:<'py>)?\s*\((?P<params>.*?)\)\s*->\s*(?P<ret>.*?)\s*\{",
        re.DOTALL,
    )

    for m in pattern.finditer(src):
        name = m.group("name")
        sig_defaults = {}
        if m.group("sig"):
            for part in split_top_level(m.group("sig")):
                part = part.strip()
                if "=" in part:
                    pname, default = part.split("=", 1)
                    sig_defaults[pname.strip()] = default.strip()
                else:
                    sig_defaults[part] = None

        params = split_top_level(m.group("params"))
        py_params = []
        for p in params:
            p = p.strip()
            if not p or p.startswith("py:") or p.startswith("py :") or ":" not in p:
                continue
            pname, ptype = p.split(":", 1)
            pname, ptype = pname.strip(), ptype.strip()
            pytype = rust_type_to_py(ptype)
            default = sig_defaults.get(pname)
            if default is not None:
                if default in ("true", "false"):
                    default_py = default.capitalize()
                elif re.match(r'^".*"$', default) or re.match(r"^-?[\d.]+$", default):
                    default_py = default
                elif default == "None":
                    default_py = "None"
                    pytype = pytype if "None" in pytype else pytype + " | None"
                else:
                    default_py = "..."
                py_params.append(f"{pname}: {pytype} = {default_py}")
            else:
                # Required: either no defaults on this function, or listed
                # in #[pyo3(signature=...)] with no `=value` because a
                # later param in the same signature has one.
                py_params.append(f"{pname}: {pytype}")

        ret_py = ret_type_to_py(m.group("ret"))

        before = src[: m.start()]
        docstring = None
        doc_search = re.search(r"((?:^///[^\n]*\n)+)$", before, re.MULTILINE)
        if doc_search:
            lines = [
                l[4:] if l.startswith("/// ") else l[3:]
                for l in doc_search.group(1).strip("\n").split("\n")
            ]
            docstring = "\n".join(lines)

        funcs.append((name, py_params, ret_py, docstring))

    return funcs


def main():
    lib_funcs = parse_file(HERE / "src" / "lib.rs")
    cloud_funcs = parse_file(HERE / "src" / "cloud.rs")
    all_funcs = sorted(lib_funcs + cloud_funcs, key=lambda f: f[0])

    out = [
        '"""Type stubs for the surtgis PyO3 extension module.',
        "",
        "Generated from the #[pyfunction] signatures in src/lib.rs and",
        "src/cloud.rs. Regenerate with gen_stubs.py after adding/changing a",
        "binding rather than hand-editing this file out of sync with the",
        "Rust source.",
        '"""',
        "",
        "import typing",
        "",
        "import numpy as np",
        "import numpy.typing as npt",
        "",
    ]
    for name, params, ret, doc in all_funcs:
        sig = ", ".join(params)
        out.append(f"def {name}({sig}) -> {ret}:")
        if doc:
            esc_doc = doc.replace('"""', "'''")
            if esc_doc.endswith('"'):
                esc_doc += " "
            out.append(f'    """{esc_doc}')
            out.append('    """')
        out.append("    ...")
        out.append("")

    print("\n".join(out))
    print(f"# Generated {len(all_funcs)} function stubs", file=sys.stderr)


if __name__ == "__main__":
    main()
