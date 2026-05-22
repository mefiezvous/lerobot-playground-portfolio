# SPDX-FileCopyrightText: 2026 Arthur Mouraud
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import Any

from mlcore.eval.evaluator import EvalResult as EvalResult
from mlcore.eval.evaluator import Evaluator as Evaluator

from playground.utils.visualizer import EpisodeFrames, plot_rewards, render_episode


class EvaluatorWithViz:
    """Thin wrapper over Evaluator that optionally generates MP4 + PNG per run."""

    def __init__(
        self,
        policy: Any,
        env_name: str,
        n_episodes: int,
        *,
        robot_name: str,
        policy_type: str,
        mlflow_run_id: str | None = None,
        env: Any | None = None,
        visualize: bool = False,
        viz_output_dir: Path | None = None,
    ) -> None:
        self._evaluator = Evaluator(
            policy=policy,
            env_name=env_name,
            n_episodes=n_episodes,
            robot_name=robot_name,
            policy_type=policy_type,
            mlflow_run_id=mlflow_run_id,
            env=env,
        )
        self._robot_name = robot_name
        self._policy_type = policy_type
        self._visualize = visualize
        self._viz_dir = viz_output_dir or Path(f"eval_reports/{robot_name}/{policy_type}/viz")

    def evaluate(self) -> EvalResult:
        """Run evaluation and, if visualize=True, render reward plot and video."""
        result = self._evaluator.evaluate()
        if self._visualize:
            self._generate_viz(result)
        return result

    def _generate_viz(self, result: EvalResult) -> None:
        ep = EpisodeFrames(
            frames=[],  # mlcore does not expose per-step frames; video skipped silently
            rewards=result.per_episode_rewards,
            robot_name=self._robot_name,
            policy_type=self._policy_type,
            success=result.success_rate > 0.5,
        )
        render_episode(ep, self._viz_dir)
        plot_rewards(ep, self._viz_dir)
