import socket
from pathlib import Path

PATHS = {
    "magic": {
        "data": "/data/tai/phd/data",
        "training": "/data/tai/phd/training",
        "results": "/data/tai/phd/results",
    },
    "hercules.digitalsense.local": {
        "data": "/home/data/tai/phd/data",
        "training": "/home/data/tai/phd/training",
        "results": "/home/data/tai/phd/results",
    },
}


def get_path(*args):
    hostname = socket.gethostname()
    assert all([arg in PATHS[hostname].keys() for arg in args]), "Args must be in {}".format(PATHS[hostname].keys())
    paths = tuple([Path(PATHS[hostname][arg]) for arg in args])
    return paths[0] if len(paths) == 1 else paths


def prepare_training_dir(subfolder=None, prefix="exp_"):
    base_dir = get_path("training")
    if subfolder is not None:
        base_dir = base_dir / subfolder

    previous_experiments = [int(l.stem.split('_')[1]) for l in base_dir.glob(f'{prefix}*')]
    last_experiment = max(previous_experiments) if len(previous_experiments) > 0 else 0

    # Reuse last experiment number if the folder is empty
    out_path = base_dir / f"{prefix}{last_experiment:04d}"
    logs_dir = out_path / "logs"
    ckpts_dir = out_path / "ckpts"
    if logs_dir.exists() and ckpts_dir.exists() and len(list(logs_dir.glob("*"))) == 0 and len(list(ckpts_dir.glob("*"))) == 0:
        return logs_dir, ckpts_dir

    # Else create a new directory
    out_path = base_dir / f"{prefix}{last_experiment + 1:04d}"
    logs_dir = out_path / "logs"
    ckpts_dir = out_path / "ckpts"
    logs_dir.mkdir(exist_ok=True, parents=True)
    ckpts_dir.mkdir(exist_ok=True, parents=True)

    return logs_dir, ckpts_dir
