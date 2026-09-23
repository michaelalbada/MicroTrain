"""Terminal presentation for microTrain's long-running curriculum."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table


STAGE_LABELS = {
    "base": "Base",
    "sft": "SFT",
    "dpo": "DPO",
    "rlvr": "RLVR",
    "agent_rl": "Agent RL",
}


def _percent(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _optional_percent(value: object | None) -> str:
    return "—" if value is None else _percent(value)


def _training_stats(stage: str, metrics: dict[str, float | int]) -> str:
    if stage == "sft":
        return f"loss {metrics['loss']:.3f}"
    if stage == "dpo":
        return f"loss {metrics['loss']:.3f}  pref {_percent(metrics['preference_accuracy'])}"
    return (
        f"reward {metrics['reward']:.3f}  "
        f"correct {_percent(metrics['correct_rate'])}  "
        f"tool {_percent(metrics['tool_use_rate'])}/"
        f"{_percent(metrics['tool_attempt_rate'])} exec/attempt  "
        f"KL {metrics['kl']:.3g}"
    )


class Reporter:
    """Small Rich-based UI; training and evaluation artifacts stay JSONL."""

    def __init__(self, *, console: Console | None = None, json_mode: bool = False) -> None:
        self.console = console or Console()
        self.json_mode = json_mode
        self.completed: list[str] = []

    def _completed(self, label: str) -> None:
        if label not in self.completed:
            self.completed.append(label)

    def header(
        self,
        *,
        command: str,
        model: str,
        run_dir: Path,
        device: str,
        dtype: str,
        smoke: bool,
    ) -> None:
        if self.json_mode:
            return
        mode = "smoke" if smoke else "full"
        subtitle = f"{command}  •  {device} / {dtype}  •  {mode}  •  {run_dir}"
        self.console.print(
            Panel(
                f"[bold]microTrain[/bold]  [dim]{model}[/dim]\n"
                "[cyan]base → SFT → DPO → RLVR → agent RL → eval[/cyan]",
                subtitle=subtitle,
                border_style="cyan",
                padding=(0, 2),
                expand=True,
            )
        )

    def prerequisites(self, target: str, actions: Sequence[str]) -> None:
        if self.json_mode or not actions:
            return
        labels = []
        for action in actions:
            if action == "data":
                labels.append("Data")
            elif action.startswith("eval:"):
                labels.append(f"Evaluate {STAGE_LABELS[action.removeprefix('eval:')]}")
            else:
                labels.append(STAGE_LABELS[action])
        chain = "  →  ".join(labels)
        self.console.print(
            Panel(
                f"[yellow]{chain}[/yellow]\n[dim]Running these first, then continuing with {target}.[/dim]",
                title="Automatic prerequisites",
                border_style="yellow",
                expand=False,
            )
        )

    def section(self, label: str) -> None:
        if not self.json_mode:
            self.console.rule(f"[bold cyan]{label}[/bold cyan]", align="left")

    def success(self, label: str, detail: str = "") -> None:
        self._completed(label)
        if self.json_mode:
            return
        self.console.print(f"[green]✓[/green] [bold]{label}[/bold]")
        if detail:
            self.console.print(f"  [dim]{detail}[/dim]")

    def warning(self, label: str, detail: str) -> None:
        if self.json_mode:
            return
        self.console.print(
            Panel(detail, title=f"[bold yellow]{label}[/bold yellow]", border_style="yellow")
        )

    @contextmanager
    def training(
        self, stage: str, total: int
    ) -> Iterator[Callable[[dict[str, float | int]], None]]:
        if self.json_mode:
            yield lambda metrics: None
            return
        label = STAGE_LABELS[stage]
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[stats]}[/dim]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        with progress:
            task_id = progress.add_task(f"Train {label}", total=total, stats="starting")

            def update(metrics: dict[str, float | int]) -> None:
                progress.update(
                    task_id,
                    completed=int(metrics["step"]),
                    stats=_training_stats(stage, metrics),
                )

            yield update
            progress.update(task_id, completed=total)

    def training_summary(self, stage: str, metrics: dict[str, float | int]) -> None:
        if self.json_mode:
            return
        steps = int(metrics["steps"])
        if stage == "sft":
            detail = f"mean loss {metrics['mean_loss']:.3f}"
        elif stage == "dpo":
            detail = (
                f"mean loss {metrics['mean_loss']:.3f}  •  "
                f"preference {_percent(metrics['mean_preference_accuracy'])}"
            )
        else:
            detail = (
                f"mean reward {metrics['mean_reward']:.3f}  •  "
                f"correct {_percent(metrics['mean_correct_rate'])}  •  "
                f"tool attempts {_percent(metrics['mean_tool_attempt_rate'])}  •  "
                f"tool executions {_percent(metrics['mean_tool_use_rate'])}  •  "
                f"zero-variance groups "
                f"{_percent(metrics['mean_zero_variance_groups'])}  •  "
                f"KL {metrics['mean_kl']:.3g}"
            )
        self.console.print(f"[bold]Aggregate[/bold]  {steps} updates  •  {detail}")

    @contextmanager
    def evaluation(
        self, stage: str, examples: int
    ) -> Iterator[Callable[[int, int, str], None]]:
        if self.json_mode:
            yield lambda completed, total, phase: None
            return
        label = STAGE_LABELS[stage]
        progress = Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[phase]}[/dim]"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        total = max(1, examples)
        with progress:
            task_id = progress.add_task(
                f"Evaluate {label}", total=total, phase="generations"
            )

            def update(completed: int, updated_total: int, phase: str) -> None:
                phase_total = max(1, updated_total // 2)
                phase_completed = (
                    completed - phase_total if phase == "preferences" else completed
                )
                progress.update(
                    task_id,
                    completed=phase_completed,
                    total=phase_total,
                    phase=phase,
                )

            yield update
            progress.update(task_id, completed=total)

    def evaluation_result(self, metrics: dict[str, object]) -> None:
        label = STAGE_LABELS[str(metrics["stage"])]
        self._completed(f"{label} evaluation")
        if self.json_mode:
            return
        self.console.print(
            f"[green]✓[/green] [bold]{label}[/bold]  "
            f"quality [bold cyan]{_percent(metrics['quality_index'])}[/bold cyan]  "
            f"direct [bold]{_percent(metrics['direct_accuracy'])}[/bold]  "
            f"tool [bold]{_percent(metrics['tool_accuracy'])}[/bold]  "
            f"format {_percent(metrics['valid_format_rate'])}  "
            f"preference {_percent(metrics['preference_accuracy'])}  "
            f"calls {_percent(metrics['tool_use_rate'])}/"
            f"{_percent(metrics['tool_attempt_rate'])} exec/attempt"
        )

    def evaluations(self, records: Sequence[dict[str, object]]) -> None:
        if self.json_mode:
            payload: object = records[0] if len(records) == 1 else list(records)
            self.console.file.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            self.console.file.flush()
            return
        if not records:
            return
        table = Table(
            title="Capability progression",
            box=box.SIMPLE_HEAD,
            header_style="bold cyan",
            show_edge=False,
        )
        table.add_column("Stage", style="bold")
        table.add_column("Quality", justify="right", style="bold cyan")
        table.add_column("Direct", justify="right")
        table.add_column("Tool", justify="right")
        table.add_column("Attempt", justify="right")
        table.add_column("Execute", justify="right")
        table.add_column("Format", justify="right")
        table.add_column("Preference", justify="right")
        table.add_column("Easy", justify="right")
        table.add_column("Medium", justify="right")
        table.add_column("Hard", justify="right")
        table.add_column("Unneeded tool", justify="right")
        for record in records:
            by_difficulty = record["by_difficulty"]
            assert isinstance(by_difficulty, dict)
            table.add_row(
                STAGE_LABELS[str(record["stage"])],
                _percent(record["quality_index"]),
                _percent(record["direct_accuracy"]),
                _percent(record["tool_accuracy"]),
                _optional_percent(record.get("tool_attempt_rate")),
                _optional_percent(record.get("tool_use_rate")),
                _percent(record["valid_format_rate"]),
                _percent(record["preference_accuracy"]),
                _percent(by_difficulty["easy"]["tool_accuracy"]),  # type: ignore[index]
                _percent(by_difficulty["medium"]["tool_accuracy"]),  # type: ignore[index]
                _percent(by_difficulty["hard"]["tool_accuracy"]),  # type: ignore[index]
                _percent(record["unnecessary_tool_rate"]),
            )
        self.console.print()
        self.console.print(table)

    def artifacts(self, run_dir: Path, *, plot: bool) -> None:
        if self.json_mode:
            return
        paths = [
            f"checkpoints  {run_dir / 'checkpoints'}",
            f"metrics      {run_dir / 'metrics.jsonl'}",
            f"generations  {run_dir / 'eval_<stage>.jsonl'}",
            f"trajectories {run_dir / 'train_<stage>_trajectories.jsonl'}",
        ]
        if plot:
            paths.append(f"plot         {run_dir / 'capability_curve.svg'}")
        self.console.print(Panel("\n".join(paths), title="Artifacts", border_style="green"))

    def finish(self, run_dir: Path, *, elapsed_seconds: float) -> None:
        if self.json_mode:
            return
        seconds = max(0, round(elapsed_seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            elapsed = f"{hours}h {minutes}m {seconds}s"
        elif minutes:
            elapsed = f"{minutes}m {seconds}s"
        else:
            elapsed = f"{seconds}s"

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="bold")
        summary.add_column()
        summary.add_row("Completed", ", ".join(self.completed) or "Command completed")
        summary.add_row("Elapsed", elapsed)
        summary.add_row("Run", str(run_dir))
        self.console.print(
            Panel(
                summary,
                title="[bold green]Run complete[/bold green]",
                border_style="green",
                expand=False,
            )
        )
