# SnartConnect AI — Project Plan

**Goal:** Train a small neural network via self-play (AlphaZero-style) for the SnartConnect Connect-4 variant, and run it in the browser at playable strength with a download under ~5 MB.

**Constraint:** The existing game is in JavaScript, running in the browser. The AI must integrate with it and be served as static assets.

---

## Architecture Overview

```
snartconnect/
├── engine/                    # Shared game logic (JS, browser + Node)
│   ├── state.js               # Board representation, move application
│   ├── rules.js               # Win/draw/terminal detection
│   ├── zobrist.js             # Hashing for transposition + repetition
│   └── engine.test.js         # Golden test suite (see Appendix A)
│
├── ai/                        # Browser-side AI
│   ├── mcts.js                # MCTS with PUCT, used at inference
│   ├── inference.js           # ONNX runtime wrapper
│   └── model.onnx             # Trained model (served static)
│
├── train/                     # Python training pipeline
│   ├── engine_py/             # Python port of engine/ (mirror)
│   │   ├── state.py
│   │   ├── rules.py
│   │   └── zobrist.py
│   ├── mcts.py                # Python MCTS (self-play)
│   ├── network.py             # PyTorch model definition
│   ├── self_play.py           # Parallel self-play workers
│   ├── replay_buffer.py       # Training data storage
│   ├── train_loop.py          # Orchestrates self-play → train → eval
│   ├── evaluate.py            # Gating: new net vs old net
│   ├── export_onnx.py         # Export for browser
│   └── config.yaml            # Hyperparameters
│
├── shared/
│   └── tests/                 # JSON test vectors, consumed by both langs
│
├── web/                       # Existing SnartConnect UI
│   └── ai-integration.js      # Hooks AI into the game UI
│
└── tools/
    ├── run_training.sh
    └── benchmark_engine.js
```

---

## Milestones

### M0: Engine Specification & Test Suite

**Deliverable:** A frozen `shared/tests/positions.json` file specifying engine behavior.

**Why first:** The engine is the ground truth for both training and browser play. Any bug here poisons everything downstream. Both the JS and Python engines will pass the same tests.

**Work:**

1. Define the exact rules:
   - Board dimensions (confirm 7×6 or whatever yours is).
   - Move semantics: drop, overflow shift, bottom ejection.
   - Win condition: 4-in-a-row, and points mode (4/5/6/7 = 1/2/5/10, first to 10).
   - Draw conditions: turn cap (start with 300 plies) and threefold repetition.
2. Write 50–100 hand-crafted test positions in JSON, each with:
   - Board state
   - Player to move
   - Legal moves
   - Expected terminal status (if any)
   - Zobrist hash
3. Include edge cases:
   - Overflow creating a win.
   - Overflow destroying a win.
   - Same state reached via two paths (repetition).
   - Points-mode scoring.

**Acceptance:** Test file reviewed and frozen. No further rule changes without a version bump.

---

### M1: JavaScript Engine (Browser-Compatible)

**Deliverable:** `engine/` module that passes all M0 tests, runs in browser and Node, and is fast.

**Work:**

1. `state.js`:
   - Represent board as `Uint8Array(42)` (values 0/1/2).
   - Track: current player, move count, history of hashes (for repetition), points totals.
   - `clone()` must be cheap. Consider copy-on-write or a small immutable wrapper.
2. `rules.js`:
   - `legalMoves(state) -> [0..6]` — all columns always legal in your variant, so this is trivial. Confirm.
   - `applyMove(state, col) -> newState` — must handle shift + eject correctly.
   - `isTerminal(state) -> {done, winner, reason}`.
   - `evaluateWin(state, player) -> bool` for each of the 4 directions.
3. `zobrist.js`:
   - Precompute 42×2 random 64-bit keys as BigInt.
   - Hash includes current player bit.
   - Incremental update on move (XOR out/in).
4. `engine.test.js`:
   - Load `shared/tests/positions.json`, run each, assert.
   - Run in Node (`node engine.test.js`) and in browser (open an HTML file).

**Acceptance:**

- All M0 tests pass in both runtimes.
- `applyMove` benchmarks at < 1 µs per call on your dev machine.
- No DOM dependencies in `engine/`.

---

### M2: Python Engine Mirror

**Deliverable:** `train/engine_py/` passing the same M0 tests.

**Why:** Self-play needs to run fast and parallel. Python with NumPy is fine; we don't need to match JS perf, but we need identical behavior.

**Work:**

1. Port `state.py`, `rules.py`, `zobrist.py` line-by-line from JS.
2. Add `test_engine_py.py` that consumes the same JSON test file.
3. Use `numpy.uint8` for the board; `int` for the hash (Python ints are arbitrary precision, so no overflow concerns).

**Acceptance:** All M0 tests pass. Parity with JS confirmed by running both against the same random playouts and comparing hashes.

**Parity test (important):** Generate 10,000 random games in JS, dump the full state history to JSON. Replay in Python, assert hash equality at every step. This catches subtle bugs.

---

### M3: MCTS (No Network) — Baseline

**Deliverable:** A working MCTS in JS that plays reasonably by random playouts.

**Why:** Before adding a network, verify the MCTS plumbing. A pure MCTS baseline is also a useful opponent for evaluation later.

**Work:**

1. `ai/mcts.js` (and mirrored in `train/mcts.py`):
   - Standard MCTS with UCT or PUCT.
   - Rollouts to terminal state, respecting the turn cap.
   - Transposition table keyed by Zobrist hash — **critical** for your loopy game.
   - Handle the loopy nature: cap search depth and use the turn-cap rule for rollouts.
2. Configurable simulation count.
3. Simple CLI: `node play.js --sims 1000` plays against itself and prints the game.

**Acceptance:**

- MCTS with 1000 sims beats your existing deterministic mode in a 20-game match.
- MCTS with 1000 sims beats MCTS with 100 sims (sanity check).

---

### M4: Network & Self-Play Loop (Proof of Concept)

**Deliverable:** A trained model that beats pure MCTS. This is your proof-of-concept milestone.

**Work:**

1. `train/network.py` (PyTorch):
   - Input: 4 planes of 7×6:
     - Plane 0: current player's pieces
     - Plane 1: opponent's pieces
     - Plane 2: all-ones if current player is P1, else zeros (turn indicator)
     - Plane 3: column heights normalized (helps the network reason about shift risk)
   - Body: 2–4 residual blocks, 64 channels.
   - Policy head → 7 logits.
   - Value head → tanh scalar.
2. `train/mcts.py` — upgrade to use the network:
   - PUCT with network policy priors.
   - Value from network at leaf nodes.
   - Dirichlet noise at root during self-play.
   - Temperature schedule (1.0 → 0.1).
3. `train/self_play.py`:
   - Multiprocessing with one worker per CPU core.
   - Each worker runs N games, records (state, policy, value) tuples.
   - Save to disk as `.npz` shards.
4. `train/replay_buffer.py`:
   - Sliding window of recent iterations (e.g., last 20).
   - Sample minibatches.
5. `train/train_loop.py`:
   - Loop: self-play → train (few epochs) → evaluate → checkpoint.
   - Gating: new net must win ≥ 55% against old net over 100 games with 200 sims each.
6. `train/evaluate.py`:
   - Head-to-head match runner with configurable sims and openings.

**Initial hyperparameters:**

- MCTS sims during self-play: 100 (start low, raise later)
- MCTS sims during gating: 200
- Batch size: 256
- Learning rate: 1e-3, decay on plateau
- Replay window: 20 iterations
- Games per iteration: 200–500

**Acceptance:**

- Training loss decreases over first 20 iterations.
- After ~50 iterations, network+MCTS (200 sims) beats pure MCTS (1000 sims) with > 60% win rate.
- No training collapse (loss doesn't diverge).

---

### M5: ONNX Export & Browser Inference

**Deliverable:** Model running in the browser, playing against a human, at interactive speed.

**Work:**

1. `train/export_onnx.py`:
   - Export with `torch.onnx.export`, opset 17.
   - Verify with `onnx.checker`.
   - Optionally quantize to FP16 or INT8; test strength after.
2. `ai/inference.js`:
   - Load `onnxruntime-web` with WebGL backend.
   - Wrap model in a `predict(boardTensor) -> {policy, value}` function.
   - Handle model loading async, with a loading spinner.
3. `ai/mcts.js` — extend to accept the network:
   - PUCT with network priors.
   - Value at leaves from network.
   - Configurable sims (start with 200–400 for browser).
4. `web/ai-integration.js`:
   - Add "Play vs AI" mode to the existing UI.
   - Difficulty slider → maps to sims count (e.g., 50 / 200 / 800).
   - Show "thinking…" indicator while MCTS runs.
   - Run MCTS in a Web Worker so the UI doesn't freeze.

**Acceptance:**

- Model + runtime download ≤ 5 MB gzipped.
- First inference after load < 50 ms.
- 200-sim move decision < 1 second on a mid-range laptop.
- AI beats your deterministic mode > 90% of games.

---

### M6: Long Training & Tuning

**Deliverable:** A strong model, tuned for browser play.

**Work:**

1. Scale self-play: 1000+ games per iteration, 400 sims.
2. Increase network to 8 blocks / 128 channels if M5 model plateaus.
3. Tune:
   - Dirichlet epsilon and alpha.
   - Temperature schedule.
   - Replay window size.
   - Learning rate schedule.
4. Consider adding:
   - Symmetry augmentation (horizontal flip of the board — valid for your game? Confirm; the shift mechanic may or may not preserve symmetry, but the board layout does).
   - Auxiliary targets (e.g., predict final score margin in points mode).
5. Track everything: use `tensorboard` or `wandb`.

**Acceptance:**

- Model beats M5 model > 70% in head-to-head.
- Elo progression plotted and monotonic.

---

## Cross-Cutting Concerns

### Parallelism

- **Self-play:** `multiprocessing.Pool` in Python, one worker per physical core.
- **Training:** single GPU (P100). Batch size limited by memory; start small.
- **Browser MCTS:** Web Worker, one per AI move. Don't block the UI.

### Reproducibility

- Every training run gets a config hash and a seed.
- Save: config, git commit, model checkpoint, replay buffer snapshot.
- `tools/run_training.sh` takes a config path and runs the full loop.

### Testing Gates

- M0 test file is frozen. Any change requires a version bump and re-running all engines.
- Every milestone has acceptance criteria that must pass before moving on.
- No milestone proceeds on a failing previous milestone.

### Performance Budgets

| Component | Budget |
|---|---|
| Engine `applyMove` (JS) | < 1 µs |
| Engine `applyMove` (Python) | < 10 µs |
| Network forward pass (browser, WebGL) | < 5 ms |
| Network forward pass (P100, batch 256) | < 50 ms |
| MCTS 200 sims (browser) | < 1 s |
| Self-play 1 game, 100 sims | < 2 s per core |
| Total browser download (gzip) | < 5 MB |

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Engine bug poisons training | Medium | M0 test suite + JS/Python parity test |
| Self-play too slow | High | Parallel workers, fewer sims early, DAG transposition table |
| Training collapse | Medium | Gating, replay window, LR decay |
| Model too large for browser | Low | See size analysis; cap at 4 blocks / 64 ch initially |
| Loopy MCTS doesn't converge | Medium | Turn cap, depth cap, transposition table |
| ONNX runtime incompatibility | Low | Test onnxruntime-web early in M5 |
| Python/JS engine drift | Medium | Shared test file + parity hash test |

---

## Appendix A: Test Vector Format

```json
{
  "version": 1,
  "dimensions": { "cols": 7, "rows": 6 },
  "tests": [
    {
      "id": "basic-drop",
      "board": "000000000000000000000000000000000000000000",
      "currentPlayer": 1,
      "points": { "p1": 0, "p2": 0 },
      "history": [],
      "move": 3,
      "expectedBoard": "000000000000000000000000000000000000000100",
      "expectedHash": "0x...",
      "expectedTerminal": null
    },
    {
      "id": "overflow-creates-win",
      "comment": "Dropping into a full column shifts pieces and creates a win",
      "board": "...",
      "move": 2,
      "expectedTerminal": { "winner": 1, "reason": "connect4" }
    },
    {
      "id": "threefold-repetition-draw",
      "comment": "History contains two prior occurrences of this exact state",
      "history": ["0x...", "0x...", "0x..."],
      "expectedTerminal": { "winner": 0, "reason": "repetition" }
    }
  ]
}
```

---

## Appendix B: Recommended First Commit Sequence

1. `engine/state.js` + `engine/rules.js` + a basic smoke test.
2. `shared/tests/positions.json` with 10 tests, growing to 100.
3. `engine/zobrist.js` + hash tests.
4. `train/engine_py/` mirror + parity test.
5. `ai/mcts.js` (random rollouts).
6. `train/network.py` + a single forward pass test.
7. `train/mcts.py` (network-guided).
8. `train/self_play.py` for 1 game, 1 worker.
9. `train/train_loop.py` end-to-end for 5 iterations.
10. Everything after that is scaling.

---

## Appendix C: What Qwen Should Do First

When you start coding with Qwen, give it this order:

1. "Write `engine/state.js` and `engine/rules.js` per the spec above."
2. "Write 10 test positions in `shared/tests/positions.json` covering basic drop, overflow, win, and repetition."
3. "Write `engine/engine.test.js` to run them."
4. "Write the Python mirror."
5. Proceed in milestone order.

Don't skip M0. Don't let Qwen write the MCTS before the engine is frozen and tested. The engine is the foundation; everything else is replaceable.

---

## Appendix D: Model Size Analysis

| Network config | Params | FP32 size | FP16 size | INT8 size |
|---|---|---|---|---|
| 4 blocks / 64 ch | ~310k | ~1.3 MB | ~650 KB | ~330 KB |
| 8 blocks / 128 ch | ~1.2M | ~5 MB | ~2.5 MB | ~1.3 MB |
| 20 blocks / 256 ch | ~15M | ~60 MB | ~30 MB | ~15 MB |

**Recommendation:** Ship the 4 blocks / 64 ch model in FP32 ONNX. Total browser download (model + onnxruntime-web + engine) will be ~3 MB uncompressed, ~1.5 MB gzipped.

---

*End of plan.*
