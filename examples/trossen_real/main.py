"""Robot-client entrypoint for deploying TA-VLA pi0 on the real Trossen bimanual arms.

Run this in the ``~/lerobot_trossen/.venv`` (py3.12 + openpi-client), from the repo root, while
``scripts/serve_policy.py`` is serving the trained checkpoint in the JAX ``~/EVAN-TA-VLA/.venv``:

    cd ~/EVAN-TA-VLA && ~/lerobot_trossen/.venv/bin/python -m examples.trossen_real.main \
        --host localhost --port 8000 --action_horizon 25 --dry-run

Drop ``--dry-run`` to actually command the arms. See ``examples/trossen_real/README.md``.
"""

import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.trossen_real import env as _env


@dataclasses.dataclass
class Args:
    # Websocket policy server (openpi scripts/serve_policy.py).
    host: str = "localhost"
    port: int = 8000

    # Steps executed per inference chunk before re-querying the server. The model predicts a
    # 50-step chunk; re-inferring more often (25) keeps the effort history and images fresh.
    action_horizon: int = 25

    # Control loop rate. Must match the 30 fps the dataset was recorded / trained at so the
    # effort-history frame offsets (-36..0) span the same wall-clock window (~1.2 s).
    max_hz: float = 30.0

    num_episodes: int = 1
    max_episode_steps: int = 1000

    # Language prompt injected into every observation. MUST byte-match the training default_prompt.
    prompt: str = _env.DEFAULT_PROMPT

    # Safety: per-step relative joint-target clamp enforced by the follower (rad). None disables.
    max_relative_target: float | None = 1.0
    # Optional action smoothing in absolute joint space; 1.0 = off.
    action_ema_alpha: float = 1.0

    # If set, assemble observations and query the server but never command the arms.
    dry_run: bool = False

    # Optional hardware overrides (default to $FOLLOWER_*_IP_ADDR / $CAM_*_SN env vars).
    left_arm_ip: str | None = None
    right_arm_ip: str | None = None
    cam_high_serial: str | None = None
    cam_left_wrist_serial: str | None = None
    cam_right_wrist_serial: str | None = None


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    environment = _env.TrossenRealEnvironment(
        left_arm_ip=args.left_arm_ip,
        right_arm_ip=args.right_arm_ip,
        cam_high_serial=args.cam_high_serial,
        cam_left_wrist_serial=args.cam_left_wrist_serial,
        cam_right_wrist_serial=args.cam_right_wrist_serial,
        prompt=args.prompt,
        max_relative_target=args.max_relative_target,
        action_ema_alpha=args.action_ema_alpha,
        dry_run=args.dry_run,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        environment.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Parse the dataclass directly (not `tyro.cli(main)`) so flags are `--host`/`--port`/… rather
    # than `--args.host` — newer tyro nests a function's dataclass parameter under its name.
    main(tyro.cli(Args))
