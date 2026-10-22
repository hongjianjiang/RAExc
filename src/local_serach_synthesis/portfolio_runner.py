import time
import random
import copy
import math
from multiprocessing import Process, Queue, set_start_method

# --- Utilities ---------------------------------------------------------------

def _seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass

def _snapshot(learner):
    """Return a serializable snapshot of the learner's current best."""
    return {
        "best_score": float(getattr(learner, "best_score", 0.0)),
        "dra_state": learner.save_dra_state(),  # serializable dict
    }

def _apply_dra_state(learner, dra_state: dict):
    """Load a saved DRA state back into the learner (in-place)."""
    learner.load_dra_state(dra_state)

# --- Individual algorithm adapters ------------------------------------------

def _run_hill_climbing(learner, timeout, args):
    # print("Starting Hill Climbing (HC)...")
    learner.hill_climbing(timeout=timeout, restart_iterations=args.get("restart_iterations", 10_000))
    # print(f"Finished HC. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_simulated_annealing(learner, timeout, args):
    # print("Starting Simulated Annealing (SA)...")

    learner.simulated_annealing(
        initial_temperature=args.get("initial_temperature", 1.0),
        cooling_rate=args.get("cooling_rate", 0.99),
        max_iterations=args.get("max_iterations", 10_000),
        timeout=timeout
    )
    # print(f"Finished SA. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_aco(learner, timeout, args):
    # print("Starting Ant Colony Optimization (ACO)...")
    learner.ant_colony_optimization(
        n_ants=args.get("n_ants", 10),
        n_iterations=args.get("n_iterations", 2_000),
        evaporation_rate=args.get("evaporation_rate", 0.5),
        alpha=args.get("alpha", 1.0),
        beta=args.get("beta", 2.0),
        timeout=timeout
    )
    # print(f"Finished ACO. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_lns(learner, timeout, args):
    # print("Starting Large Neighborhood Search (LNS)...")
    learner.large_neighborhood_search(
        n_iterations=args.get("n_iterations", 2_000),
        destruction_size=args.get("destruction_size", 2),
        timeout=timeout
    )
    # print(f"Finished LNS. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_pso(learner, timeout, args):
    # print("Starting Particle Swarm Optimization (PSO)...")
    learner.particle_swarm_optimization(
        swarm_size=args.get("swarm_size", 10),
        n_iterations=args.get("n_iterations", 2_000),
        timeout=timeout
    )
    # print(f"Finished PSO. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_ga(learner, timeout, args):
    # print("Starting Genetic Algorithm (GA)...")
    learner.genetic_algorithm(
        population_size=args.get("population_size", 16),
        generations=args.get("generations", 2_000),
        mutation_rate=args.get("mutation_rate", 0.3),
        timeout=timeout
    )
    # print(f"Finished GA. Best score: {learner.best_score}")
    return _snapshot(learner)

def _run_tabu(learner, timeout, args):
    # print("Starting Tabu Search (TS)...")
    learner.tabu_search(
        n_iterations=args.get("n_iterations", 2_000),
        tabu_size=args.get("tabu_size", 50),
        timeout=timeout
    )
    # print(f"Finished TS. Best score: {learner.best_score}")
    return _snapshot(learner)

ALGO_REGISTRY = {
    "hill_climbing": _run_hill_climbing,
    "sim_anneal": _run_simulated_annealing,
    "aco": _run_aco,
    "lns": _run_lns,
    "pso": _run_pso,
    "ga": _run_ga,
    "tabu": _run_tabu,
}

# --- Parallel portfolio runner ----------------------------------------------

def _worker_entry(fn_name, base_learner, timeout, args, seed, out_q):
    try:
        _seed_everything(seed)
        # Deep-copy learner to isolate mutations/state
        learner = copy.deepcopy(base_learner)
        start = time.time()
        res = ALGO_REGISTRY[fn_name](learner, timeout, args or {})
        res["algo"] = fn_name
        res["elapsed"] = time.time() - start
        out_q.put(res)
    except Exception as e:
        out_q.put({"algo": fn_name, "error": repr(e), "best_score": -1.0})

def run_portfolio_parallel(
    learner,
    per_algo_timeout: float = 300.0,
    algos: list[str] = None,
    algo_args: dict | None = None,
    base_seed: int = 42,
    start_method: str = "spawn",
):
    """
    Run multiple algorithms in parallel processes, each with its own timeout.

    Returns:
        dict with keys:
          - "best_algo": str
          - "best_score": float
          - "best_state": dra_state dict
          - "results": list of per-algo dicts
    """
    if algos is None:
        algos = list(ALGO_REGISTRY.keys())
    if algo_args is None:
        algo_args = {}

    # On some systems, you must set start method exactly once at the entry point.
    try:
        set_start_method(start_method)
    except RuntimeError:
        pass  # already set

    q = Queue()
    procs: list[Process] = []
    for i, name in enumerate(algos):
        p = Process(
            target=_worker_entry,
            args=(
                name,
                learner,
                per_algo_timeout,
                algo_args.get(name, {}),
                base_seed + i,
                q,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    results = []
    for _ in procs:
        res = q.get()  # Blocks until each worker reports
        results.append(res)

    for p in procs:
        try:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
        except Exception:
            pass

    # Pick the global best
    best = max(results, key=lambda r: r.get("best_score", -1.0))
    return {
        "best_algo": best.get("algo"),
        "best_score": best.get("best_score", 0.0),
        "best_state": best.get("dra_state"),
        "results": results,
    }

# --- Anytime racing (round-robin with slices) -------------------------------

def run_portfolio_racing(
    learner,
    total_budget: float = 600.0,
    time_slice: float = 10.0,
    algos: list[str] = None,
    algo_args: dict | None = None,
    base_seed: int = 42,
):
    """
    Run an anytime 'racing' schedule: each algorithm gets short repeated slices.
    Good when algos benefit from many brief restarts.

    Returns the same dict as run_portfolio_parallel.
    """
    if algos is None:
        algos = list(ALGO_REGISTRY.keys())
    if algo_args is None:
        algo_args = {}

    # Keep a private clone per algorithm
    states = {name: copy.deepcopy(learner) for name in algos}
    scores = {name: float(getattr(learner, "best_score", 0.0)) for name in algos}
    best_state = learner.save_dra_state()
    best_algo = None
    start = time.time()

    # Assign deterministic but distinct seeds per algo
    for i, name in enumerate(algos):
        _seed_everything(base_seed + 1000 + i)

    # Round-robin until budget exhausted
    idx = 0
    while time.time() - start < total_budget:
        name = algos[idx % len(algos)]
        worker = ALGO_REGISTRY[name]
        ler = states[name]
        t0 = time.time()
        try:
            # Each slice runs the algorithm with a small timeout
            res = worker(ler, timeout=time_slice, args=algo_args.get(name, {}))
            sc = float(res["best_score"])
            if sc > scores[name]:
                scores[name] = sc
            if sc > float(_snapshot(learner)["best_score"]):
                # Keep a global best snapshot
                best_algo = name
                best_state = res["dra_state"]
        except Exception:
            pass  # ignore failures this slice
        idx += 1

    # Build per-algo result list shaped like the parallel one
    results = []
    for name in algos:
        results.append({
            "algo": name,
            "best_score": scores[name],
            "dra_state": states[name].save_dra_state(),
            "elapsed": None,  # variable across slices
        })

    # If no algo improved beyond the original, pick the top scorer anyway
    if best_algo is None:
        best_algo = max(scores.items(), key=lambda kv: kv[1])[0]
        best_state = states[best_algo].save_dra_state()

    return {
        "best_algo": best_algo,
        "best_score": scores[best_algo],
        "best_state": best_state,
        "results": results,
    }
