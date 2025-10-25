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
import os, csv, json, time, shlex, subprocess, re, numpy as np

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
    UT_HOST: str       = "glg1"            # target machine alias (NOT the gateway IP)
    SUT_USER: str       = "gl_greenest"     # lab user
    SUT_KEY:  Path = Path.home() / ".ssh/id_ed25519"
    SUT_WORKDIR: Path  = Path("~/greenestlab")        # where sut_trial.py + prompts live on SUT
    SUT_PYTHON: str    = "python3"
    PROMPTS_CSV: str   = "prompts_mmlu_subset.csv" # on SUT
    QUESTIONS_PER_TRIAL: int = 20

    # Jump host (gateway)
    SUT_JUMP_HOST: str  = "145.108.225.3"
    SUT_JUMP_PORT: int  = 42224
    SUT_JUMP_USER: str  = "gl_greenest"

    # local temp file to capture last trial rows for populate_run_data()
    _last_rows_file = None

    # Dynamic configurations can be one-time satisfied here before the program takes the config as-is
    # e.g. Setting some variable based on some criteria



    def __init__(self):
        """Executes immediately after program start, on config load"""
        # --- SSH / SUT settings (instance attributes so subprocess sees them) ---
        # Target compute node (NOT the gateway)
        self.SUT_HOST       = os.getenv("SUT_HOST", "glg1")           # target node alias
        self.SUT_USER       = os.getenv("SUT_USER", "gl_greenest")
        self.SUT_KEY        = Path(os.getenv("SUT_KEY", str(Path.home()/".ssh/id_ed25519")))
        self.SUT_JUMP_HOST  = os.getenv("SUT_JUMP_HOST", "145.108.225.3")
        self.SUT_JUMP_PORT  = int(os.getenv("SUT_JUMP_PORT", "42224"))
        self.SUT_JUMP_USER  = os.getenv("SUT_JUMP_USER", "gl_greenest")


        # Where Ollama models live (on SUT); just tags mapping:
        self.MODEL_TAGS = {
            "Llama-3 8B Instruct": {"fp16": "llama3:8b-instruct-fp16", "q8_0": "llama3:8b-instruct-q8_0", "q4_0": "llama3:8b-instruct-q4_0"},
            "Mistral-7B Instruct": {"fp16": "mistral:7b-instruct-fp16", "q8_0": "mistral:7b-instruct-q8_0", "q4_0": "mistral:7b-instruct-q4_0"},
            "GLM-4 9B Chat": {"fp16": "glm4:9b-chat-fp16", "q8_0": "glm4:9b-chat-q8_0", "q4_0": "glm4:9b-chat-q4_0"},
        }
        # MMLU subset CSV path on SUT
        self.PROMPTS_CSV = os.getenv("PROMPTS_CSV", "prompts_mmlu_subset.csv")

        # Trial size (same questions each trial)
        self.QUESTIONS_PER_TRIAL = int(os.getenv("QUESTIONS_PER_TRIAL", "25"))

        # EnergiBridge CLI settings (server)
        self.EB_BIN        = os.getenv("EB_BIN", "energi_bridge")
        self.EB_INTERVALMS = os.getenv("EB_INTERVALMS", "200")
        self.EB_USE_GPU    = os.getenv("EB_USE_GPU", "1") == "1"

        # internal
        self._last_rows_file = None
        
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
    
    def _ssh_base(self):
        base = ["ssh","-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=accept-new",
                "-J", f"{self.SUT_JUMP_USER}@{self.SUT_JUMP_HOST}:{self.SUT_JUMP_PORT}"]
        if self.SUT_KEY and self.SUT_KEY.exists():
            base += ["-i", str(self.SUT_KEY)]
        return base

    def _ssh_run(self, remote_cmd: str, stdin_text: str = None, timeout: int = 60*60):
        target = f"{self.SUT_USER}@{self.SUT_HOST}"
        cmd = self._ssh_base() + [target, remote_cmd]
        return subprocess.run(cmd, input=stdin_text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    def _scp_from_sut(self, remote_path: str, local_dir: Path):
        base = ["scp","-O","-o","BatchMode=yes","-o","StrictHostKeyChecking=accept-new",
                "-J", f"{self.SUT_JUMP_USER}@{self.SUT_JUMP_HOST}:{self.SUT_JUMP_PORT}"]
        if self.SUT_KEY and self.SUT_KEY.exists():
            base += ["-i", str(self.SUT_KEY)]
        remote = f"{self.SUT_USER}@{self.SUT_HOST}:{remote_path}"
        local_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(base + ["-r", remote, str(local_dir)], check=True)

    def _eb_cli(self, action: str):
        """Send 'start', 'stop', or 'collect' to the EnergiBridge service on the SUT."""
        remote = f"energibridge {action}"
        r = self._ssh_run(remote)
        if r.returncode != 0:
            print(f"[WARN] EnergiBridge {action} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def _eb_start(self, trial_id: str, out_dir: str = None):
        eb_out = out_dir or f"/tmp/eb_{trial_id}"
        gpu_flag = "--gpu" if self.EB_USE_GPU else ""
        # Detach with nohup; store PID for later stop
        remote = (
            f"mkdir -p {shlex.quote(eb_out)}; "
            f"nohup {shlex.quote(self.EB_BIN)} record {gpu_flag} --interval {shlex.quote(self.EB_INTERVALMS)} "
            f"--out {shlex.quote(eb_out)} > /dev/null 2>&1 & echo $! > /tmp/eb_{trial_id}.pid"
        )
        r = self._ssh_run(remote)
        if r.returncode != 0:
            raise RuntimeError(f"EB start failed: {r.stderr}")
        return eb_out

    def _eb_stop(self, trial_id: str):
        remote = "PID=$(cat /tmp/eb_{tid}.pid 2>/dev/null); " \
                "if [ -n \"$PID\" ]; then kill $PID 2>/dev/null; fi; " \
                "pkill -f {bin} 2>/dev/null; rm -f /tmp/eb_{tid}.pid".format(
                    tid=trial_id, bin=shlex.quote(self.EB_BIN))
        r = self._ssh_run(remote)
        # ignore non-zero; pkill may have nothing to kill

    def _ollama_run(self, tag: str, prompt: str):
        """
        Run Ollama on the SUT with a plain-text prompt (no --json available).
        Returns (text, meta_dict, wall_time_s).
        """
        # Prepare the remote command (no --json)
        remote = f"ollama run {shlex.quote(tag)}"
        t0 = time.time()

        # Send the prompt through stdin
        res = self._ssh_run(remote_cmd=remote, stdin_text=prompt, timeout=60*15)
        t1 = time.time()

        if res.returncode != 0:
            raise RuntimeError(f"Ollama failed ({tag}): {res.stderr}")

        # Everything comes as plain text now
        text_out = res.stdout.strip()
        total_s = t1 - t0

        # No token stats available in this mode
        meta = {"eval_count": 0, "eval_duration": 0}

        return text_out, meta, total_s
    
    LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)

    def _extract_letter(self, txt: str) -> str:
        """
        Extracts the first multiple-choice letter (A/B/C/D) from a model's response.
        Falls back to scanning text if not cleanly formatted.
        """
        if not txt:
            return ""
        m = self.LETTER_RE.search(txt.upper())
        if m:
            return m.group(1).upper()
        # fallback: take the first A/B/C/D anywhere in text
        for ch in txt.upper():
            if ch in "ABCD":
                return ch
        return ""

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
        This is invoked before measurement is started.  We use this phase to warm up the
        system so that the GPU and CPU temperatures stabilise before we begin energy
        measurement.  The Green Lab checklist requires a warm‑up period to stabilise the
        temperature【257292169943939†L204-L209】.  We implement a temperature‑based warm‑up
        strategy: measure the current GPU temperature, send a dummy prompt to the model
        repeatedly until the temperature rises above the baseline by a small delta, and
        record that baseline for later cool‑off.  This avoids using a fixed wait time and
        instead adapts based on the system state.
        """
        # Ensure the per-run folder exists
        (context.run_dir).mkdir(parents=True, exist_ok=True)

        # Determine the model tag for this run to warm up the correct model
        model = context.execute_run["model"]
        quant = context.execute_run["quantization_level"]
        tag   = self.MODEL_TAGS[model][quant]

        # Record the baseline GPU temperature before any work
        baseline_temp = self._get_remote_temperature()
        self._baseline_temp = baseline_temp

        # Perform warm‑up by sending dummy prompts until temperature rises above baseline
        self._warm_up(tag, baseline_temp)

        # After warm‑up the system should be warmed; measurement will start in start_measurement


    def start_measurement(self, context: RunnerContext) -> None:
        """Perform any activity required for starting measurements."""
        # Start energy measurement on the remote SUT.  The energibridge service on the
        # cluster exposes start/stop/collect commands.  We invoke start here, after
        # warm‑up has completed.  If the command fails, a warning is logged and
        # measurement results will be zero.
        self._eb_cli("start")

    def interact(self, context):
        model = context.execute_run["model"]
        quant = context.execute_run["quantization_level"]
        tag   = self.MODEL_TAGS[model][quant]

        # Run prompts without starting/stopping measurement here.  The measurement is
        # started in start_measurement and will be stopped in stop_measurement.  We
        # simply send prompts to the model, collect predictions and latencies, and
        # store them for aggregation later.
        rows = []
        for r in self._load_prompts():
            prompt = self._format_prompt(r)
            text, meta, wall = self._ollama_run(tag, prompt)
            letter = self._extract_letter(text)
            gold = r["answer"].strip().upper()
            rows.append({
                "gold": gold,
                "pred": letter,
                "correct": int(letter == gold),
                "latency_s": wall
            })

        # Save per‑prompt results to run directory; energy will be added in stop_measurement
        self._save_rows(context.run_dir, rows)


    def stop_measurement(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping measurements."""
        # Stop energy measurement and collect the results from the remote EB service
        eb_json = self._eb_cli("collect")
        self._eb_cli("stop")

        # Parse energy summary.  If parsing fails (e.g., EB not available), values remain zero.
        cpu_j = gpu_j = 0.0
        try:
            data = json.loads(eb_json)
            cpu_j = float(data.get("cpu_joules", 0.0))
            gpu_j = float(data.get("gpu_joules", 0.0))
        except Exception:
            print("[WARN] Could not parse EB CLI output during stop_measurement")

        # Read previously saved rows and append energy information.  We compute per‑prompt
        # energy by dividing total Joules by the number of prompts.
        if self._last_rows_file and self._last_rows_file.exists():
            rows = json.loads(self._last_rows_file.read_text(encoding="utf-8"))
            if rows:
                per_prompt_energy = (cpu_j + gpu_j) / max(1, len(rows))
                for rr in rows:
                    rr["energy_cpu_j"] = cpu_j / max(1, len(rows))
                    rr["energy_gpu_j"] = gpu_j / max(1, len(rows))
                    rr["energy_per_token_j"] = None  # tokens not available in plain mode
                    rr["energy/prompt"] = per_prompt_energy
                # Write updated rows back to file
                with open(self._last_rows_file, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)

    def stop_run(self, context: RunnerContext) -> None:
        """Perform any activity here required for stopping the run.
        Activities after stopping the run should also be performed here."""
        # After measurement has been stopped, perform a cool‑off period.  We use the
        # baseline temperature recorded during warm‑up and wait until the GPU
        # temperature drops close to that baseline.  This satisfies the cool‑down
        # requirement of the checklist【257292169943939†L204-L209】 without relying on
        # a fixed sleep time.
        baseline_temp = getattr(self, "_baseline_temp", None)
        if baseline_temp is not None:
            self._cool_off(baseline_temp)
            # Reset baseline so it doesn’t interfere with subsequent runs
            self._baseline_temp = None

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
        if not self._last_rows_file or not self._last_rows_file.exists():
            return None
        rows = json.loads(self._last_rows_file.read_text(encoding="utf-8"))
        if not rows:
            return None

        lat  = np.array([r["latency_s"]       for r in rows], dtype=float)
        tps  = np.array([r["tokens_per_s"]    for r in rows], dtype=float)
        acc  = np.array([r["correct"]         for r in rows], dtype=float)
        ecpu = np.array([r["energy_cpu_j"]    for r in rows], dtype=float)
        egpu = np.array([r["energy_gpu_j"]    for r in rows], dtype=float)

        # trial-level scalar copied to rows
        ept_vals = [r.get("energy_per_token_j") for r in rows]
        ept = float(np.nanmean([v for v in ept_vals if isinstance(v, (int,float))]))

        return {
            "energy/prompt":  float((ecpu + egpu).mean()),
            "energy/token":   ept,
            "accuracy":       float(acc.mean()),
            "latency/prompt": float(np.nanmean(lat)),
            "token/s":        float(np.nanmean(tps)),
            "VRAM":           float("nan"),   # not available without GPU CSV parsing; can be added later
            "RAM":            float("nan"),
            "OOM":            0,
        }

    def after_experiment(self) -> None:
        """Perform any activity required after stopping the experiment here
        Invoked only once during the lifetime of the program."""
        pass

    # ================================ Temperature-based Warm-up and Cool-off ================================
    def _get_remote_temperature(self) -> float:
        """
        Retrieve the current GPU temperature on the SUT by calling nvidia-smi.  If the command
        fails or returns no value, 0.0 is returned.  This helper can be extended to
        include CPU temperature by reading thermal zones if needed.  Using temperature
        allows us to decide when warm‑up and cool‑off are complete without a fixed delay.
        """
        cmd = "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader"
        res = self._ssh_run(cmd)
        if res.returncode != 0:
            return 0.0
        try:
            temp_str = res.stdout.strip().split("\n")[0]
            return float(temp_str)
        except Exception:
            return 0.0

    def _warm_up(self, tag: str, baseline_temp: float) -> None:
        """
        Warm up the model by repeatedly sending a dummy prompt until the GPU temperature
        rises above the baseline by at least 2°C.  This helps to stabilise the
        temperature before measurements start.  If the command fails or temperature
        cannot be read, the warm‑up loop will run a single dummy prompt.
        """
        try:
            current_temp = self._get_remote_temperature()
            # If temperature reading fails, run one dummy prompt and return
            if current_temp == 0.0:
                self._ollama_run(tag, "Say: READY")
                return
            # Warm up loop: send dummy prompts until temperature difference threshold met
            while current_temp < baseline_temp + 2.0:
                # Send a dummy prompt to warm up GPU/CPU
                try:
                    self._ollama_run(tag, "Say: READY")
                except Exception as e:
                    print(f"[WARN] Warm-up dummy prompt failed: {e}")
                    break
                # Sleep briefly to allow temperature to update
                time.sleep(1)
                current_temp = self._get_remote_temperature()
        except Exception as e:
            print(f"[WARN] Warm-up encountered an error: {e}")

    def _cool_off(self, baseline_temp: float) -> None:
        """
        Cool off the system after measurement.  Polls the GPU temperature and waits
        until it falls within 1°C of the baseline temperature measured before the run.
        This method prevents the next run from starting while the system is still hot.
        """
        try:
            current_temp = self._get_remote_temperature()
            if current_temp == 0.0:
                return
            while current_temp > baseline_temp + 1.0:
                time.sleep(2)
                current_temp = self._get_remote_temperature()
        except Exception as e:
            print(f"[WARN] Cool-off encountered an error: {e}")

    # ================================ Prompt handling ================================
    def _load_prompts(self) -> List[Dict[str, str]]:
        """Load prompts from the local CSV file on the controller.  The first
        QUESTIONS_PER_TRIAL rows are used for each run.  Each row must contain
        subject, question, A, B, C, D, answer columns."""
        prompts = []
        csv_path = self.ROOT_DIR / self.PROMPTS_CSV
        if not csv_path.exists():
            raise FileNotFoundError(f"Prompts CSV not found: {csv_path}")
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= self.QUESTIONS_PER_TRIAL:
                    break
                prompts.append(row)
        return prompts

    def _format_prompt(self, row: Dict[str, str]) -> str:
        """Format a single MMLU question row into a plain text prompt."""
        return (
            f"Question: {row['question']}\n"
            f"Options:\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n\n"
            "Respond with only one letter (A, B, C, or D)."
        )

    def _save_rows(self, run_dir: Path, rows: List[Dict[str, Any]]) -> None:
        """Save per‑prompt rows to a JSON file in the run directory for later aggregation."""
        self._last_rows_file = run_dir / "rows.json"
        with open(self._last_rows_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    
    

    # ================================ DO NOT ALTER BELOW THIS LINE ================================
    experiment_path:            Path             = None