from EventManager.Models.RunnerEvents import RunnerEvents
from EventManager.EventSubscriptionController import EventSubscriptionController
from ConfigValidator.Config.Models.RunTableModel import RunTableModel
from ConfigValidator.Config.Models.FactorModel import FactorModel
from ConfigValidator.Config.Models.RunnerContext import RunnerContext
from ConfigValidator.Config.Models.OperationType import OperationType
from ProgressManager.Output.OutputProcedure import OutputProcedure as output
from Plugins.Profilers.EnergiBridge import EnergiBridge

from typing import Dict, List, Any, Optional
from pathlib import Path
from os.path import dirname, realpath
import json, shlex, subprocess, time
import numpy as np


class RunnerConfig:
    ROOT_DIR = Path(dirname(realpath(__file__)))

    # ================================ USER SPECIFIC CONFIG ================================
    """The name of the experiment."""
    name:                       str             = "new_runner_experiment"

    """The path in which Experiment Runner will create a folder with the name `self.name`, in order to store the
    results from this experiment. (Path does not need to exist - it will be created if necessary.)
    Output path defaults to the config file's path, inside the folder 'experiments'"""
    results_output_path:        Path             = ROOT_DIR / 'experiments'

    """Experiment operation type. Unless you manually want to initiate each run, use `OperationType.AUTO`."""
    operation_type:             OperationType   = OperationType.AUTO

    """The time Experiment Runner will wait after a run completes.
    This can be essential to accommodate for cooldown periods on some systems."""
    time_between_runs_in_ms:    int             = 1000

    # SUT SSH config (edit to your env)
    SUT_HOST: str      = "gl_greenest@145.108.225.3:42224"      # IP or hostname
    SUT_USER: str      = "gl_greenest@glg1"           # SSH user
    SUT_KEY:  Path     = ROOT_DIR / "id_ed25519"  # or None to use agent/default
    SUT_WORKDIR: Path  = Path("~/greenestlab")        # where sut_trial.py + prompts live on SUT
    SUT_PYTHON: str    = "python3"
    PROMPTS_CSV: str   = "prompts_mmlu_subset.csv" # on SUT
    QUESTIONS_PER_TRIAL: int = 20

    # local temp file to capture last trial rows for populate_run_data()
    _last_rows_file = None

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria
    def __init__(self):
        """Executes immediately after program start, on config load"""

        EventSubscriptionController.subscribe_to_multiple_events([
            (RunnerEvents.BEFORE_EXPERIMENT, self.before_experiment),
            (RunnerEvents.BEFORE_RUN       , self.before_run       ),
            (RunnerEvents.START_RUN        , self.start_run        ),
            (RunnerEvents.START_MEASUREMENT, self.start_measurement),
            (RunnerEvents.INTERACT         , self.interact         ),
            (RunnerEvents.STOP_MEASUREMENT , self.stop_measurement ),
            (RunnerEvents.STOP_RUN         , self.stop_run         ),
            (RunnerEvents.POPULATE_RUN_DATA, self.populate_run_data),
            (RunnerEvents.AFTER_EXPERIMENT , self.after_experiment )
        ])
        self.run_table_model = None  # Initialized later

        output.console_log("Custom config loaded")

    def create_run_table_model(self) -> RunTableModel:
        """Create and return the run_table model here. A run_table is a List (rows) of tuples (columns),
        representing each run performed"""
        factor1 = FactorModel("model", ['Llama-3 8B Instruct', 'Mistral-7B Instruct' ,'GLM-4 9B Chat'])
        factor2 = FactorModel("quantization_level", ['fp16','q8_0','q4_0'])  # <-- updated
        self.run_table_model = RunTableModel(
            factors=[factor1, factor2],
            # exclude_combinations=[
            #     {factor2: [10]},   # all runs having treatment "10" will be excluded
            #     {factor1: ['rec'], factor2: [5000, 10000]},
            #     {factor1: ['mem', 'iter'], factor2: [35, 40]},  # all runs having the combination ("iter", 30) will be excluded
            # ],
            repetitions = 20,
            data_columns=["energy/prompt", "energy/token", "accuracy", 'latency/prompt', 'token/s', 'VRAM', 'RAM', 'OOM'],
        )
        return self.run_table_model

    def before_experiment(self) -> None:
        """Perform any activity required before starting the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    def before_run(self) -> None:
        """Perform any activity required before starting a run.
        No context is available here as the run is not yet active (BEFORE RUN)"""
        pass

    def start_run(self, context: RunnerContext) -> None:
        """Perform any activity required for starting the run here.
        For example, starting the target system to measure.
        Activities after starting the run should also be performed here."""
        #pass       
        # Create a per-run folder; ExperimentRunner usually does this, but ensure it exists
        (context.run_dir).mkdir(parents=True, exist_ok=True)

    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        fib_type = context.execute_run["model"]
        problem_size = context.execute_run["quantization_level"]
        
        # self.profiler = EnergiBridge(target_program=f"python examples/hello-world-fibonacci/fibonacci_{fib_type}.py {problem_size}",
        #                              out_file=context.run_dir / "energibridge.csv")

        # self.profiler.start()

    def interact(self, context: RunnerContext) -> None:
        """Perform any interaction with the running target system here, or block here until the target finishes."""
        #pass
        # Build and execute the SSH command to run ONE trial on SUT
        model = context.execute_run["model"]
        quant = context.execute_run["quantization_level"]

        # Optional: give each run a distinct CSV on SUT for traceability
        remote_out_csv = f"{self.SUT_WORKDIR}/trial_{int(time.time())}.csv"

        remote = (
            f"cd {shlex.quote(str(self.SUT_WORKDIR))} && "
            f"{self.SUT_PYTHON} sut_trial.py "
            f"--prompts_csv {shlex.quote(self.PROMPTS_CSV)} "
            f"--model {shlex.quote(model)} "
            f"--quant {shlex.quote(quant)} "
            f"--questions {self.QUESTIONS_PER_TRIAL} "
            f"--out_csv {shlex.quote(remote_out_csv)}"
        )
        cmd = self._ssh_cmd(remote)
        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60*60)
        t1 = time.time()
        if proc.returncode != 0:
            output.console_log(proc.stderr)
            raise RuntimeError(f"SSH run failed for {model} {quant}")

        # SUT prints one JSON payload line; parse it
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        rows = payload.get("rows", [])

        # Save rows JSON locally for populate_run_data (one file per run)
        self._last_rows_file = context.run_dir / "rows.json"
        with open(self._last_rows_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        # stdout = self.profiler.stop(wait=True)

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        pass

    def populate_run_data(self, context: RunnerContext) -> Optional[Dict[str, Any]]:
        """Parse and process any measurement data here.
        You can also store the raw measurement data under `context.run_dir`
        Returns a dictionary with keys `self.run_table_model.data_columns` and their values populated"""
        
        # eb_log, eb_summary = self.profiler.parse_log(self.profiler.logfile,
        #                                              self.profiler.summary_logfile)
        #
        # return {"energy": eb_summary["total_joules"],
        #         "runtime": eb_summary["runtime_seconds"],
        #         "memory": max(eb_log["USED_MEMORY"].values())}
        # Read rows.json and aggregate to fill data_columns
        if not self._last_rows_file or not Path(self._last_rows_file).exists():
            return None
        with open(self._last_rows_file, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if not rows:
            return None

        # arrays
        lat = np.array([r["latency_s"] for r in rows], dtype=float)
        tps = np.array([r["tokens_per_s"] for r in rows], dtype=float)
        acc = np.array([r["correct"]     for r in rows], dtype=float)
        e_cpu = np.array([r["energy_cpu_j"] for r in rows], dtype=float)
        e_gpu = np.array([r["energy_gpu_j"] for r in rows], dtype=float)

        # totals and aggregates
        accuracy          = float(acc.mean())  # 0..1 (report ×100% in writing)
        latency_per_prompt= float(lat.mean())
        tokens_per_s      = float(tps.mean())

        # If you want energy/prompt = cpu+gpu:
        energy_per_prompt = float((e_cpu + e_gpu).mean())

        # If you also track tokens generated per prompt to compute energy/token:
        # For now we estimate tokens by tps*latency; ok for a first pass
        est_tokens = (tps * lat)
        energy_per_token  = float(((e_cpu + e_gpu) / np.maximum(est_tokens, 1e-9)).mean())

        # VRAM/RAM/OOM: if not measured on SUT, return NaN / 0; or extend sut_trial.py to log them
        return {
            "energy/prompt": energy_per_prompt,
            "energy/token":  energy_per_token,
            "accuracy":      accuracy,
            "latency/prompt": latency_per_prompt,
            "token/s":        tokens_per_s,
            "VRAM":           float("nan"),
            "RAM":            float("nan"),
            "OOM":            0,
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass
    
    def _ssh_cmd(self, remote_cmd: str) -> list:
        base = ["ssh","-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new"]
        if self.SUT_KEY and Path(self.SUT_KEY).exists():
            base += ["-i", str(self.SUT_KEY)]
        target = f"{self.SUT_USER}@{self.SUT_HOST}" if self.SUT_USER else self.SUT_HOST
        return base + [target, remote_cmd]
    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None
