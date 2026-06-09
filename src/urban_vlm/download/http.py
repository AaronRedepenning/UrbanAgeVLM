from pathlib import Path

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_file(
    url: str,
    output_path: Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        print(f"Skipping existing file: {output_path}")
        return output_path

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        )

        with progress:
            task = progress.add_task(output_path.name, total=total)

            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    progress.update(task, advance=len(chunk))

    tmp_path.replace(output_path)
    return output_path
