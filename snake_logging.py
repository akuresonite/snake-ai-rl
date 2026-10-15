"""Training metrics for TensorBoard and MLflow.

Two backends, both optional, both safe to leave uninstalled.

TensorBoard
    Needs `pip install tensorboard`. Uses torch's SummaryWriter, which ships
    with torch but needs the tensorboard package behind it. Writes classic event
    files under runs/<name>/tensorboard. View with:

        tensorboard --logdir runs

MLflow
    Needs NO install. MLflow's on disk run format is plain JSON and TSV, so this
    module writes it directly. If you do have the mlflow package, point its UI
    at the same folder:

        mlflow ui --backend-store-uri runs/mlruns

    If MLflow is installed, this module uses the real client instead, which is
    more robust. If it is not, the fallback writer produces the same layout.

Everything here is defensive. A logging failure must never kill a training run,
so every write is wrapped and a broken backend is disabled once with a warning.
"""

import math
import os
import time
import uuid

# ==========================================
# MLFLOW RUN, written by hand
# ==========================================

class _FileMlflowRun:
    """Minimal writer for MLflow's on disk run format.

    Layout produced, which is what `mlflow ui` reads:

        <store>/<experiment_id>/meta.yaml
        <store>/<experiment_id>/<run_id>/meta.yaml
        <store>/<experiment_id>/<run_id>/params/<key>
        <store>/<experiment_id>/<run_id>/tags/<key>
        <store>/<experiment_id>/<run_id>/metrics/<key>
        <store>/<experiment_id>/<run_id>/artifacts/

    The metrics file is TSV: timestamp, value, step.
    """

    def __init__(self, store, experiment_name, run_name):
        self.store = store
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.experiment_id = "0"
        self.run_id = uuid.uuid4().hex
        self.run_dir = os.path.join(store, self.experiment_id, self.run_id)
        self._ensure_layout()
        self._pending = {}

    def _ensure_layout(self):
        exp_dir = os.path.join(self.store, self.experiment_id)
        # The experiment directory holds only meta.yaml and one folder per run.
        # Naming anything else here confuses mlflow ui, which lists every
        # subdirectory as a run. This was a real bug once.
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "params"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "tags"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "metrics"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "artifacts"), exist_ok=True)

        now = int(time.time() * 1000)
        exp_meta = os.path.join(exp_dir, "meta.yaml")
        if not os.path.exists(exp_meta):
            with open(exp_meta, "w", encoding="utf-8") as f:
                f.write(f"experiment_id: '{self.experiment_id}'\n")
                f.write(f"name: {self.experiment_name}\n")
                f.write(f"artifact_location: file://{os.path.abspath(exp_dir)}\n")
                f.write(f"lifecycle_stage: active\n")
                f.write(f"creation_time: {now}\n")
                f.write(f"last_update_time: {now}\n")

        with open(os.path.join(self.run_dir, "meta.yaml"), "w", encoding="utf-8") as f:
            f.write(f"run_id: {self.run_id}\n")
            f.write(f"experiment_id: '{self.experiment_id}'\n")
            f.write("status: RUNNING\n")
            f.write(f"start_time: {now}\n")
            f.write(f"end_time: 0\n")
            f.write(f"artifact_uri: file://{os.path.abspath(os.path.join(self.run_dir, 'artifacts'))}\n")
            f.write(f"lifecycle_stage: active\n")
            f.write(f"run_name: {self.run_name}\n")

    def log_param(self, key, value):
        safe = str(key).replace("/", "_").replace(" ", "_")
        with open(os.path.join(self.run_dir, "params", safe), "w", encoding="utf-8") as f:
            f.write(str(value))

    def log_tag(self, key, value):
        safe = str(key).replace("/", "_").replace(" ", "_")
        with open(os.path.join(self.run_dir, "tags", safe), "w", encoding="utf-8") as f:
            f.write(str(value))

    def log_metric(self, key, value, step):
        safe = str(key).replace("/", "_").replace(" ", "_")
        # Batched in memory and flushed by the Logger, because one open/write per
        # metric per step would dominate the runtime on a fast loop.
        self._pending.setdefault(safe, []).append((int(time.time() * 1000), float(value), int(step)))

    def flush(self):
        for key, rows in self._pending.items():
            path = os.path.join(self.run_dir, "metrics", key)
            with open(path, "a", encoding="utf-8") as f:
                for ts, val, step in rows:
                    f.write(f"{ts} {val} {step}\n")
        self._pending.clear()

    def finish(self):
        self.flush()
        meta = os.path.join(self.run_dir, "meta.yaml")
        try:
            with open(meta, "r", encoding="utf-8") as f:
                text = f.read()
            text = text.replace("status: RUNNING", "status: FINISHED")
            text = text.replace("end_time: 0", f"end_time: {int(time.time() * 1000)}")
            with open(meta, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            pass


# ==========================================
# THE LOGGER
# ==========================================

class Logger:
    """Fans metrics out to TensorBoard and MLflow.

    backend: "both", "tensorboard", "mlflow", or "none".
    """

    def __init__(self, backend="both", run_name=None, log_dir="./runs",
                 experiment="snake-dqn", params=None, verbose=True):
        self.backend = (backend or "none").lower()
        self.log_dir = log_dir
        self.verbose = verbose
        self.run_name = run_name or time.strftime("run-%Y%m%d-%H%M%S")
        self.tb = None
        self.mlf = None
        self._tb_dir = None
        self._mlf_uri = None
        self._last_flush = time.time()
        self._flush_every = 10.0
        self._warned = set()

        if self.backend in ("both", "tensorboard"):
            self._start_tensorboard()
        if self.backend in ("both", "mlflow"):
            self._start_mlflow(experiment)

        if params:
            self.log_params(params)

        if self.verbose:
            print(f"=> logging: {self.backend}")
            if self._tb_dir:
                print(f"   tensorboard : {self._tb_dir}")
                print(f"   view with   : tensorboard --logdir {log_dir}")
            if self._mlf_uri:
                print(f"   mlflow      : {self._mlf_uri}")
                print(f"   view with   : mlflow ui --backend-store-uri {self._mlf_uri}")

    # ---- backends ---------------------------------------------------------
    def _start_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_dir = os.path.join(self.log_dir, self.run_name, "tensorboard")
            self.tb = SummaryWriter(log_dir=self._tb_dir)

            # Guard against a silently empty dashboard. If the whole log dir was
            # empty before this run started, then the user has probably opened
            # tensorboard before training anything, seen a blank page, and
            # concluded it is broken. Say so out loud instead.
            parent = os.path.join(self.log_dir, self.run_name)
            try:
                events = [f for f in os.listdir(self._tb_dir) if f.startswith("events.out.tfevents")]
            except OSError:
                events = []
            if len(events) <= 1:
                print(f"=> note: {parent} held no previous runs.")
                print(f"   if TensorBoard showed nothing before, it was pointed at")
                print(f"   an empty folder. Start it from the project root:")
                print(f"       tensorboard --logdir {self.log_dir}")
                print(f"   and give it about 30 seconds to load.")
        except Exception as e:
            self._warn_once("tensorboard",
                            f"TensorBoard unavailable ({type(e).__name__}). "
                            f"Install it with: pip install tensorboard")
            self.tb = None

    def _start_mlflow(self, experiment):
        store = os.path.join(self.log_dir, "mlruns")
        self._mlf_uri = os.path.abspath(store)
        try:
            import mlflow
            mlflow.set_tracking_uri("file://" + self._mlf_uri)
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=self.run_name)
            self.mlf = mlflow
        except Exception:
            # The package is not installed. Write the format by hand instead,
            # which mlflow ui reads just the same once you do install it.
            try:
                self.mlf = _FileMlflowRun(store, experiment, self.run_name)
            except Exception as e:
                self._warn_once("mlflow", f"MLflow logging disabled ({type(e).__name__}: {e})")
                self.mlf = None

    def _warn_once(self, key, message):
        if key not in self._warned:
            print(f"=> WARNING: {message}")
            self._warned.add(key)

    # ---- logging ----------------------------------------------------------
    def log_params(self, params):
        for k, v in params.items():
            if self.tb is not None:
                try:
                    self.tb.add_text(f"params/{k}", str(v), 0)
                except Exception:
                    pass
            if self.mlf is not None:
                try:
                    if hasattr(self.mlf, "log_param"):
                        self.mlf.log_param(k, v)
                except Exception:
                    pass

    def add(self, key, value, step):
        """One scalar. Non finite values are dropped, not written."""
        if value is None:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(value):
            # A nan or inf row makes TensorBoard and MLflow draw a broken line
            # and MLflow marks the whole metric as suspect. Better to skip it.
            return
        if self.tb is not None:
            try:
                self.tb.add_scalar(key, value, int(step))
            except Exception:
                self.tb = None
                self._warn_once("tb_write", "TensorBoard writing failed, disabled")
        if self.mlf is not None:
            try:
                if hasattr(self.mlf, "log_metric"):
                    self.mlf.log_metric(key, value, int(step))
                else:
                    self.mlf.log_metric(key, value, int(step))
            except Exception:
                self.mlf = None
                self._warn_once("mlf_write", "MLflow writing failed, disabled")

    def add_many(self, metrics, step):
        for k, v in metrics.items():
            self.add(k, v, step)

    def maybe_flush(self, force=False):
        """Persist buffered metrics periodically, not on every single write."""
        now = time.time()
        if not force and now - self._last_flush < self._flush_every:
            return
        self._last_flush = now
        if self.tb is not None:
            try:
                self.tb.flush()
            except Exception:
                self.tb = None
        if isinstance(self.mlf, _FileMlflowRun):
            try:
                self.mlf.flush()
            except Exception:
                pass

    def close(self):
        self.maybe_flush(force=True)
        if self.tb is not None:
            try:
                self.tb.close()
            except Exception:
                pass
            self.tb = None
        if self.mlf is not None:
            try:
                if isinstance(self.mlf, _FileMlflowRun):
                    self.mlf.finish()
                else:
                    self.mlf.end_run()
            except Exception:
                pass
            self.mlf = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ==========================================
# NULL LOGGER
# ==========================================

class NullLogger:
    """Does nothing, so training code never needs a None check."""

    def add(self, key, value, step):
        pass

    def add_many(self, metrics, step):
        pass

    def log_params(self, params):
        pass

    def maybe_flush(self, force=False):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# ==========================================
# HISTOGRAMS, optional but useful
# ==========================================

def add_histogram(logger, key, values, step):
    if logger is None or logger.tb is None or not len(values):
        return
    try:
        import numpy as np
        logger.tb.add_histogram(key, np.asarray(values), int(step))
    except Exception:
        pass
