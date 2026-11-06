"""Phase 5 campaign supervisor: one process, N independent worker subprocesses.

The supervisor never scores anything itself.  It enumerates the frozen job list,
starts one ``phase5_runner.py run --job <id>`` process per job, keeps at most
``--workers`` alive, retries a failed job from a fresh process at most twice, and
stops the campaign on a third failure so a defect is never averaged away by reruns.

Every completed job is re-verified (SHA-256 of its array file against the ``.done``
record) before it is counted, so "complete" cannot mean "process exited 0".

State is written atomically to ``state/supervisor_<campaign>.json``; the final
per-job manifest is ``runs/<campaign>/campaign_manifest.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import phase5_common as C

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "phase5_runner.py"
sys.path.insert(0, str(HERE))
import phase5_runner as R  # noqa: E402  (job enumeration lives with the runner)


def priority(spec: dict) -> tuple[int, str]:
    """Longest-pole-first, but preps before anything that needs them."""
    kind, group = spec["kind"], spec.get("group", spec.get("variant", ""))
    if kind == "prep":
        return (0, spec["id"])
    if kind == "h0":
        return ({"anmf": 1, "hybrid": 3, "cold": 4, "ica": 5, "nophys": 4,
                 "cfar": 5, "temporal": 6, "fairness": 6}.get(group, 7), spec["id"])
    return ({"anmf": 2}.get(group, 6), spec["id"])


def worker_env() -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    env["PHASE5_TORCH_THREADS"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def env_record(cfg: dict) -> dict:
    import platform
    import numpy
    import scipy
    import torch
    return {
        "generated": C.utc_now(),
        "campaign": cfg["campaign"],
        "config_sha256": cfg["config_sha256"],
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "blas_threads_per_worker": 1,
        "torch_threads_per_worker": 1,
        "cpu_count": os.cpu_count(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="", help="comma-separated group filter (debug)")
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--budget-minutes", type=float, default=0.0,
                    help="stop starting new jobs after this many minutes (0 = no limit)")
    args = ap.parse_args(argv)

    cfg = C.config()
    run = C.run_dir(cfg)
    state_path = C.WORK / "state" / f"supervisor_{cfg['campaign']}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    env_path = run / "environment.json"
    if not env_path.is_file():
        C.atomic_write_json(env_path, env_record(cfg))
    env = worker_env()

    only = {g.strip() for g in args.only.split(",") if g.strip()}
    specs = []
    for s in R.job_specs(cfg):
        if only and not ({s["kind"], s.get("group"), s.get("variant")} & only):
            continue
        specs.append(s)
    specs.sort(key=priority)

    done: dict[str, dict] = {}
    failed: dict[str, int] = {}
    locked: dict[str, int] = {}
    running: dict[str, dict] = {}
    t_start = time.perf_counter()
    logdir = run / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    sup_log = open(logdir / "supervisor.log", "a", encoding="utf-8", buffering=1)

    def say(msg: str) -> None:
        line = f"[{C.utc_now()}] {msg}"
        print(line, flush=True)
        sup_log.write(line + "\n")

    say(f"campaign {cfg['campaign']} workers={args.workers} jobs={len(specs)} "
        f"config={cfg['config_sha256'][:12]}")

    queue = [s for s in specs if not R.is_done(run, s, cfg)[0]]
    for s in specs:
        ok, why = R.is_done(run, s, cfg)
        if ok:
            done[s["id"]] = {"status": "pre-existing"}
    say(f"already complete: {len(done)}  pending: {len(queue)}")

    stop_reason = "complete"
    while queue or running:
        while queue and len(running) < args.workers:
            if args.budget_minutes and (time.perf_counter() - t_start) / 60 > args.budget_minutes:
                stop_reason = "budget"
                break
            spec = queue.pop(0)
            logf = open(logdir / (spec["id"].replace("|", "__") + ".out"), "ab")
            proc = subprocess.Popen([sys.executable, str(RUNNER), "run", "--job", spec["id"]],
                                    cwd=str(HERE), env=env, stdout=logf,
                                    stderr=subprocess.STDOUT)
            running[spec["id"]] = {"proc": proc, "spec": spec, "log": logf,
                                   "t0": time.perf_counter()}
            say(f"start {spec['id']}")
        if not running:
            break
        time.sleep(2.0)
        for jid in list(running):
            rec = running[jid]
            rc = rec["proc"].poll()
            if rc is None:
                continue
            rec["log"].close()
            wall = time.perf_counter() - rec["t0"]
            ok, why = R.is_done(run, rec["spec"], cfg)
            if rc == 3:
                # another live process holds the job lock: requeue without penalty
                locked[jid] = locked.get(jid, 0) + 1
                say(f"locked {jid} (attempt {locked[jid]})")
                if locked[jid] <= 20:
                    queue.append(rec["spec"])
                else:
                    failed[jid] = failed.get(jid, 0) + 1
            elif rc == 0 and ok:
                done[jid] = {"wall_s": round(wall, 1), "status": "ok"}
                say(f"done  {jid} wall={wall:.0f}s")
            else:
                failed[jid] = failed.get(jid, 0) + 1
                say(f"FAIL  {jid} rc={rc} verify={why} attempt={failed[jid]}")
                if failed[jid] > args.max_retries:
                    stop_reason = f"job {jid} failed {failed[jid]} times ({why})"
                    queue.clear()
                else:
                    queue.insert(0, rec["spec"])
            del running[jid]

        payload = {
            "campaign": cfg["campaign"], "updated": C.utc_now(),
            "n_jobs": len(specs), "n_done": len(done), "n_running": len(running),
            "n_pending": len(queue), "n_failed_jobs": len(failed),
            "elapsed_min": round((time.perf_counter() - t_start) / 60, 1),
            "stop_reason": stop_reason if (not queue and not running) else None,
            "running": sorted(running), "failed": failed,
        }
        try:
            C.atomic_write_json(state_path, payload)
        except OSError as exc:          # never let telemetry kill the campaign
            say(f"state write failed: {exc}")
        if stop_reason != "complete" and not queue:
            break

    elapsed = time.perf_counter() - t_start
    if not queue and not running and stop_reason == "complete":
        rows = []
        for s in specs:
            ok, why = R.is_done(run, s, cfg)
            rec = {}
            npz, donef = R.out_paths(run, s)
            if donef.is_file():
                rec = C.read_json(donef)
            rows.append({
                "job_id": s["id"], "kind": s["kind"], "group": s.get("group"),
                "fold": s.get("fold"), "verified": ok, "why": why,
                "sha256": rec.get("sha256"), "wall_s": rec.get("wall_s"),
                "peak_rss_mb": rec.get("peak_rss_mb"), "n_units": rec.get("n_units"),
                "arrays": rec.get("arrays"),
            })
        manifest = {"campaign": cfg["campaign"], "config_sha256": cfg["config_sha256"],
                    "generated": C.utc_now(), "n_jobs": len(rows),
                    "n_verified": sum(r["verified"] for r in rows),
                    "total_wall_s_of_jobs": sum(r["wall_s"] or 0 for r in rows),
                    "supervisor_wall_s": round(elapsed, 1), "jobs": rows}
        C.atomic_write_json(run / "campaign_manifest.json", manifest)
        say(f"manifest written: {manifest['n_verified']}/{len(rows)} verified, "
            f"supervisor wall {elapsed / 3600:.2f} h")

    payload = {"campaign": cfg["campaign"], "updated": C.utc_now(),
               "n_jobs": len(specs), "n_done": len(done), "n_pending": len(queue),
               "n_failed_jobs": len(failed), "failed": failed,
               "elapsed_min": round(elapsed / 60, 1), "stop_reason": stop_reason}
    C.atomic_write_json(state_path, payload)
    say(f"supervisor exit: reason={stop_reason} done={len(done)}/{len(specs)} "
        f"failed={len(failed)} wall={elapsed / 60:.1f} min")
    return 0 if (not failed and not queue and not running) else 1


if __name__ == "__main__":
    raise SystemExit(main())
