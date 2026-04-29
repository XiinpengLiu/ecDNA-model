"""Small output helpers for fit artifacts.

The fitting code writes CSV/JSON/NPZ unconditionally and writes optional
parquet-style artifacts only when the local environment already supports them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def write_json(path: str | Path, payload: object) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")


def write_rows_csv(path: str | Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    destination = Path(path)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = tuple(keys)
    try:
        import pandas as pd  # type: ignore

        dataframe = pd.DataFrame([{field: row.get(field) for field in fieldnames} for row in rows], columns=list(fieldnames))
        dataframe.to_csv(destination, index=False)
    except Exception:
        import csv

        with open(destination, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(fieldnames))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field) for field in fieldnames})


def write_optional_parquet(path: str | Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    destination = Path(path)
    try:
        import pandas as pd  # type: ignore

        dataframe = pd.DataFrame(list(rows))
        dataframe.to_parquet(destination, index=False)
        return {"path": str(destination), "written": True, "format": "parquet"}
    except Exception as exc:
        marker = destination.with_suffix(destination.suffix + ".SKIPPED.json")
        write_json(
            marker,
            {
                "path": str(destination),
                "written": False,
                "reason": f"optional parquet support unavailable: {type(exc).__name__}: {exc}",
                "csv_fallback": str(destination.with_suffix(".csv")),
            },
        )
        return {"path": str(destination), "written": False, "reason": str(exc)}


def write_table_bundle(output_dir: str | Path, stem: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str] | None = None) -> dict[str, object]:
    destination = Path(output_dir)
    csv_path = destination / f"{stem}.csv"
    write_rows_csv(csv_path, rows, fieldnames)
    parquet_status = write_optional_parquet(destination / f"{stem}.parquet", rows)
    return {"csv": str(csv_path), "parquet": parquet_status}


def write_npz_or_marker(path: str | Path, arrays: Mapping[str, np.ndarray], *, label: str) -> dict[str, object]:
    destination = Path(path)
    try:
        np.savez(destination, **{name: np.asarray(value) for name, value in arrays.items()})
        return {"path": str(destination), "written": True, "label": label}
    except Exception as exc:
        marker = destination.with_suffix(destination.suffix + ".SKIPPED.json")
        write_json(marker, {"path": str(destination), "written": False, "label": label, "reason": str(exc)})
        return {"path": str(destination), "written": False, "label": label, "reason": str(exc)}


def write_netcdf_skip_marker(path: str | Path, *, npz_fallback: str | Path, label: str) -> None:
    write_json(
        Path(path).with_suffix(Path(path).suffix + ".SKIPPED.json"),
        {
            "path": str(path),
            "written": False,
            "label": label,
            "reason": "NetCDF writing failed; see write_netcdf_file return status.",
            "npz_fallback": str(npz_fallback),
        },
    )


def _write_xarray_netcdf(destination: Path, arrays: Mapping[str, np.ndarray], *, label: str) -> None:
    import xarray as xr  # type: ignore

    data_vars = {}
    for name, value in arrays.items():
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            array = array.reshape(1)
        dims = tuple(f"{name}_dim{axis}" for axis in range(array.ndim))
        data_vars[name] = (dims, array)
    dataset = xr.Dataset(data_vars=data_vars, attrs={"history": label})
    dataset.to_netcdf(destination)


def _write_scipy_netcdf(destination: Path, arrays: Mapping[str, np.ndarray], *, label: str) -> None:
    from scipy.io import netcdf_file  # type: ignore

    with netcdf_file(str(destination), mode="w") as handle:
        handle.history = label
        for name, value in arrays.items():
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = array.reshape(1)
            dim_names = []
            for axis, size in enumerate(array.shape):
                dim = f"{name}_dim{axis}"
                handle.createDimension(dim, int(size))
                dim_names.append(dim)
            variable = handle.createVariable(name, "f8", tuple(dim_names))
            variable[:] = array


def write_netcdf_file(path: str | Path, arrays: Mapping[str, np.ndarray], *, label: str) -> dict[str, object]:
    destination = Path(path)
    try:
        _write_xarray_netcdf(destination, arrays, label=label)
        return {"path": str(destination), "written": True, "label": label, "backend": "xarray"}
    except Exception as xarray_exc:
        try:
            _write_scipy_netcdf(destination, arrays, label=label)
            return {"path": str(destination), "written": True, "label": label, "backend": "scipy.io.netcdf_file", "xarray_error": str(xarray_exc)}
        except Exception as scipy_exc:
            write_netcdf_skip_marker(destination, npz_fallback=destination.with_suffix(".npz"), label=label)
            return {"path": str(destination), "written": False, "label": label, "reason": f"xarray: {xarray_exc}; scipy: {scipy_exc}"}


def write_text_pdf(path: str | Path, title: str, lines: Iterable[str]) -> None:
    """Write a minimal one-page PDF with plain ASCII diagnostic text."""

    rendered = [title, ""] + [str(line) for line in lines]
    safe_lines = []
    for line in rendered[:48]:
        ascii_line = line.encode("latin-1", "replace").decode("latin-1")
        ascii_line = ascii_line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        safe_lines.append(ascii_line[:110])
    content = ["BT", "/F1 10 Tf", "72 760 Td"]
    first = True
    for line in safe_lines:
        if first:
            content.append(f"({line}) Tj")
            first = False
        else:
            content.append("0 -14 Td")
            content.append(f"({line}) Tj")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    with open(Path(path), "wb") as handle:
        handle.write(bytes(payload))
