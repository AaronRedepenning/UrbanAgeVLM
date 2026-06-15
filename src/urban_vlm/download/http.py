from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from rich.console import Console, Group
from rich.live import Live
from rich.markup import escape
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from urban_vlm.utils import url_filename

ProgressCallback = Callable[[int, int | None], None]


def download_path(url: str, output_dir: Path) -> Path:
    return output_dir / Path(url_filename(url)).name


def _content_length(response: requests.Response) -> int | None:
    value = response.headers.get("content-length")
    if not value:
        return None

    try:
        total = int(value)
    except ValueError:
        return None

    return total if total > 0 else None


def download_file(
    url: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    chunk_size: int = 1024 * 1024,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = download_path(url, output_dir)

    if output_path.exists() and not overwrite:
        size = output_path.stat().st_size

        if progress_callback is not None:
            progress_callback(size, size)

        return output_path

    tmp_path = output_path.with_name(output_path.name + ".part")

    with requests.get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()

        total = _content_length(response)
        downloaded = 0

        if progress_callback is not None:
            progress_callback(downloaded, total)

        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if progress_callback is not None:
                    progress_callback(downloaded, total)

    tmp_path.replace(output_path)

    if progress_callback is not None:
        progress_callback(downloaded, total)

    return output_path


def download_urls(
    urls: Iterable[str],
    output_dir: Path,
    *,
    overwrite: bool = False,
    timeout_seconds: int = 60,
    max_workers: int = 8,
    show_progress: bool = True,
    overall_description: str = "Downloading files",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[str] = []
    seen_paths: set[Path] = set()

    for url in urls:
        url = str(url).strip()

        if not url:
            continue

        path = download_path(url, output_dir)

        # Avoid two threads writing to the same file.
        if path in seen_paths:
            continue

        seen_paths.add(path)
        jobs.append(url)

    if not jobs:
        return []

    max_workers = max(1, min(max_workers, len(jobs)))

    if not show_progress:
        return _download_urls_without_progress(
            jobs,
            output_dir,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
        )

    return _download_urls_with_progress(
        jobs,
        output_dir,
        overwrite=overwrite,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        overall_description=overall_description,
    )


def _download_urls_without_progress(
    urls: list[str],
    output_dir: Path,
    *,
    overwrite: bool,
    timeout_seconds: int,
    max_workers: int,
) -> list[Path]:
    results: list[Path | None] = [None] * len(urls)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                download_file,
                url,
                output_dir,
                overwrite=overwrite,
                timeout_seconds=timeout_seconds,
            ): index
            for index, url in enumerate(urls)
        }

        for future in as_completed(future_to_index):
            index = future_to_index[future]
            results[index] = future.result()

    return [path for path in results if path is not None]


def _download_urls_with_progress(
    urls: list[str],
    output_dir: Path,
    *,
    overwrite: bool,
    timeout_seconds: int,
    max_workers: int,
    overall_description: str,
) -> list[Path]:
    console = Console()
    progress_lock = Lock()

    overall_progress = Progress(
        TextColumn("[dim]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    file_progress = Progress(
        SpinnerColumn(),
        TextColumn("[dim]{task.description}", justify="left"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    overall_task_id = overall_progress.add_task(
        escape(overall_description),
        total=len(urls),
    )

    results: list[Path | None] = [None] * len(urls)
    errors: list[tuple[str, BaseException]] = []

    def make_progress_callback(task_id: int) -> ProgressCallback:
        def callback(downloaded: int, total: int | None) -> None:
            with progress_lock:
                file_progress.update(
                    task_id,
                    completed=downloaded,
                    total=total,
                    visible=True,
                )

        return callback

    def worker(url: str, task_id: int) -> Path:
        return download_file(
            url,
            output_dir,
            overwrite=overwrite,
            timeout_seconds=timeout_seconds,
            progress_callback=make_progress_callback(task_id),
        )

    render_group = Group(overall_progress, file_progress)

    with Live(
        render_group,
        console=console,
        refresh_per_second=12,
        transient=True,
    ):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_info = {}

            for index, url in enumerate(urls):
                output_path = download_path(url, output_dir)

                task_id = file_progress.add_task(
                    escape(output_path.name),
                    total=None,
                    visible=False,
                )

                future = executor.submit(worker, url, task_id)
                future_to_info[future] = (index, url, task_id)

            for future in as_completed(future_to_info):
                index, url, task_id = future_to_info[future]

                try:
                    results[index] = future.result()
                except BaseException as exc:
                    errors.append((url, exc))
                finally:
                    with progress_lock:
                        file_progress.update(task_id, visible=False)
                        overall_progress.advance(overall_task_id)

    if errors:
        message = "\n".join(f"- {url}: {error}" for url, error in errors)

        raise RuntimeError(
            f"Failed to download {len(errors)} file(s):\n{message}"
        ) from errors[0][1]

    return [path for path in results if path is not None]
