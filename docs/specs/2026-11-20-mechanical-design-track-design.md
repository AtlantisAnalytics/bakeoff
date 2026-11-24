# Mechanical design track: design

Status: for owner approval, 2026-11-20. Build starts after the 2026-12-07 launch.

## Intent

bakeoff scores models per track. Today there is one track, `general` (GSM8K, IFEval, our own format,
JSON, grounding and needle suites). This adds the first domain track, **mechanical design**: CAD,
machining, 3D printing and component assembly, up to whole small systems that also need wiring,
controller code and a test plan.

The tasks are written by a practitioner (the owner, a mechanical engineer by background) and are
checked by machine wherever possible. The aim is a test that reflects how engineering work is really
done: changing existing parts, making parts that fit other parts, and keeping a whole system
consistent across disciplines, with safety as a hard constraint.

## Why a new track, given the prior art

2026 already has academic text-to-CAD benchmarks: Text2CAD-Bench (arXiv 2605.18430), CADTestBench
(2605.07807, test-based), P3D-BENCH (2606.11152) and BenchCAD. They mostly score one-shot "describe a
part, get code". None of them covers:

- editing an existing part, or making a part that has to fit other parts;
- consistency across disciplines (CAD, bill of materials, wiring, controller code, test plan);
- operating-environment constraints (for example heat) that limit materials in every discipline;
- safety as a gate;
- a maintained, re-run result on hardware people own.

## Goals

1. Three task kinds, built in this order:
   - **K2, edit a part:** an existing CadQuery part plus a change request ("printable without
     supports", "0.2 mm clearance on these bores").
   - **K3, fit a part:** given parts plus a requirement ("a lid for this box", "a spacer between these
     two"), make the new part.
   - **K5, capstone system:** a short system spec; the model produces CAD parts, a bill of materials,
     a wiring list, controller code and a test plan.
   K1 (a part from a written spec) appears only as the calibration anchor; K4 (a written review for
   manufacture) comes later, once the owner has written problem lists.
2. **Safety is a gate, not a score.** A task with any failed safety check fails, whatever its cost,
   weight or speed. Those other figures compare only designs that passed the gate.
3. **Same test for every model** (bakeoff's rule): no engine-specific tool calling; the feedback loop
   is plain chat turns.
4. **Re-derivable:** every round's reply and feedback is saved, and grading is deterministic from the
   saved replies, so anyone can re-grade a run.
5. **A calibration anchor:** one existing one-shot academic set, run as published, so our CAD
   numbers can be checked against the paper's. CADTestBench first; another if its data licence
   doesn't allow redistribution of a slice.

## Non-goals

- Full agentic tool use (approach B below). It may come later as a separate "open tools" mode.
- Real hardware, CAM toolpaths or G-code.
- Free-form CAD formats. Parts are CadQuery only (v1).
- Medical, reverse engineering, trading or other domains. They are parked.
- One blended score. Mechanical design results are never merged into the general track's numbers.

## Anti-goals (would make this a failure even if it works)

- A design that is unsafe but scores well on cost or speed.
- Measuring an engine's tool-call parser instead of the model.
- Executing model-written code outside a sandbox.
- Any task that reveals the owner's professional or client work. Tasks may share a general shape
  with real projects but never their specifics.

## Approaches considered

- **A. One shot, graded afterwards.** Fits today's quality pass, cheap, re-gradable. Rejected as the
  main mode: no chance to find and fix mistakes, which is the skill being measured. Kept for the
  calibration anchor, which is one-shot by design.
- **B. Full agent with real tools.** Closest to real work. Rejected for now: tool calling differs
  across vLLM, llama.cpp and TabbyAPI and across model templates, so it would partly measure the
  engine; also the most to build and the least predictable run time.
- **C. Fixed rounds of feedback (chosen by the owner, 2026-11-20).** The model replies with all its
  artifacts; the harness runs a *visible* check set and sends the report back as a plain user turn;
  the model gets a fixed number of revision rounds; *hidden* checks are graded on the final round.

## How C works

### A task

Each task is one JSONL item in a suite under `suites/mechanical/` with a manifest like any other
suite (track `mechanical`, a `split`, a licence, a version). An item holds:

- the brief (text) and any given parts (CadQuery source, as text);
- a **closed catalog** for that task: the components, materials and wires the model may choose from,
  each with the properties the checks need (for example a motor's torque and current, a
  controller's pin list, a valve's fail state, a material's or insulation's maximum service
  temperature). Choices outside the catalog fail the format check.
- for K5, **temperature zones** (regions of space with a maximum temperature), the **hazards** the
  design must handle, and each hazard's **safe state and deadline** (for example "over-temperature:
  heater power off within 2 s");
- the visible checks and the hidden checks, by name, with their parameters (the expected answers).

### The reply format

The model replies with labelled fenced blocks: one block per CadQuery part (`part=<name>`), a JSON
bill of materials, a JSON wiring list (from pin to pin, wire type), the controller code, and a JSON
test plan. K2 and K3 need only the part blocks. A reply that can't be parsed counts as a failed round
and the feedback says what was wrong with the format.

### Controller code and the simulator

Controller code is Python against a small fixed API: a class with a `step(inputs, t) -> outputs`
method called at a fixed tick, where inputs and outputs are the named pins from the model's own
wiring list. Python avoids a compiler in the sandbox. For each K5 task, a plant model (written with
the task) simulates the system: sensors, actuators, timing, and injected faults.

### Rounds

- Round 1: the brief. Rounds 2..N: the previous reply's visible-check report. N = 3 in v1 (one reply
  plus two revisions); the manifest may set it per suite.
- The visible report lists each visible check as pass or fail with a short reason, never the
  expected values of hidden checks.
- The final round's artifacts are graded on every check. Earlier rounds are saved but not scored.

### The checks

Visible and hidden checks come from one registry; each task says which are visible.

- **Builds:** every part executes and yields a valid solid.
- **Geometry:** dimensions, volume, hole positions and clearances against the task's expected values,
  with tolerances.
- **Unchanged elsewhere (K2):** outside the requested change, the part matches the original.
- **Fit (K3, K5):** placed parts don't intersect; required gaps are within tolerance.
- **Printability:** overhang angles and minimum wall thickness on the tessellated mesh, where the
  task asks for printability.
- **Consistency (K5):** every CAD part is in the bill of materials and the reverse; every wire ends on
  a pin that exists on the chosen catalog component; the controller code reads and writes only pins
  that are wired; actuators meet the stated loads; the supply covers the sum of loads.
- **Environment (K5):** each placed part's material, and each wire's insulation and each sensor's
  rating, is good for the zone it sits in.
- **Safety (K5), always hidden:**
  - in the simulator, for every hazard (over-temperature, stop button, lost sensor signal, loss of
    a heat source or flow, controller crash), the outputs reach the safe state within the deadline;
  - by design, from the wiring list and the catalog: valves and similar actuators fail to the safe
    state on power loss; the stop button breaks power directly in the circuit, not only as a
    software input; a dead controller leaves every output safe.
- **The model's own test plan (K5):** it covers every hazard in the brief, and running it in the
  simulator gives the outcomes it predicts.

### Scoring

- A task **passes** if every safety check passes and every other required check passes.
- If any safety check fails, the task fails, and its other figures are not reported as wins.
- Per-check pass rates, the bill of materials' cost and weight, and the round in which each check
  first passed are reported for information.
- Verdicts, once the track counts (see below), are per tier × track from the task pass rate. Until
  then the track is information only.

### Sandbox

Model-written code (CadQuery parts and controller code) runs only in a sandbox: no network, a
read-only root, a scratch directory, CPU and memory limits, and a wall-clock limit per execution. A
timeout or crash is a failed check with that reason. The sandbox runs CPU-only and must not disturb
the GPU measurement (no execution while a speed pass runs).

### Where it runs

The mechanical design suites run as a separate pass after the quality pass, on the same host, during
the weekend window. CadQuery and its geometry kernel are an optional install (`bakeoff[mech]`) so the
core harness keeps its single dependency.

### Privacy and publication

v1 tasks are all public. Today `publish` refuses any bundle with private-suite outputs, and the public
form of private results is not decided (see `publish.py`); private mechanical tasks wait for that
design.

### Calibration anchor

A slice of the chosen academic set runs one-shot through the same sandbox, graded the paper's way,
and is reported next to the paper's published figures for the same or nearest models. If our
numbers disagree with the paper's, that is investigated before the track's own results are trusted.

## Decided by the owner

- Domain: mechanical design (CAD, machining, 3D printing, assembly), owner-authored.
- Task kinds K2 and K3 first, aiming at the K5 capstone.
- Safety is a gate that nothing can buy down.
- Approach C, with A only for the calibration anchor and B possibly later.
- Two cross-discipline coupling families to test: knock-on effects of a component choice, and
  environment limits on materials.

## My calls (for the owner to check)

- CadQuery only, v1.
- Python controller code against a fixed `step()` API, rather than Arduino C or a PLC language.
- A closed catalog per task, so the checks are deterministic.
- N = 3 rounds.
- A task passes only if everything required passes; partial credit is information only.
- Power budget as a consistency check.
- Mechanical design runs after the quality pass on the same host, as an optional install.
- **A new suite kind with its own driver.** Multi-round tasks don't fit the single-reply
  `grade(output, expected, options)` contract. The runner gains an interactive driver and the suite
  loader a new item schema; the existing quality pass and graders are untouched.
- **Per-round limits.** `max_tokens` applies to each reply. Every round's request carries the whole
  conversation so far (brief, earlier replies, earlier reports). Token counts and time are recorded
  per round and summed per task. "Cost" in scoring always means the bill of materials' cost, never
  inference cost.
- **A reference solution for every task.** Each task ships a golden reply that must pass every check,
  plus at least one deliberately unsafe variant that must fail the safety gate; CI runs both. A task
  without them can't be added to a suite.
- **The calibration anchor gates publication.** Before the track's own results appear on the site,
  the anchor slice must land within 5 points of the paper's figure for the nearest model, or the
  difference must be explained in a published note. Its data licence is checked first; if it can't
  be redistributed, the next set on the list is used.
- **Linux tiers only in v1.** The real sandbox runs on the Linux lab hosts. CI and development on
  macOS run the checks against a mocked sandbox. The 36 GB unified (Mac) tier gets the track later.
- **Simulated time.** Hazard deadlines are in simulator time at the task's fixed tick, never wall-clock.

## Decided after the builder check (owner, 2026-11-20)

- **Who writes what.** The owner writes each brief and its expected answers, and safety-reviews every
  K5 task. The builder writes the catalogs, the plant simulations and the reference solutions from
  the owner's briefs, for the owner to review.
- **v1 size.** About 8 K2 tasks, 8 K3 tasks and 2 K5 tasks.
- **Information only at first.** While the task count is small, the track publishes pass rates and
  per-check results but never changes a pick or a verdict. It starts counting toward verdicts once it
  reaches a task count set and published in advance.
- **Public tasks only in v1.** The track launches with public tasks. How private-task results appear
  on the site is a separate design, done later; until then private mechanical tasks are not run for
  publication.

## Left to the builder

- The exact reply-format grammar and parser.
- The sandbox mechanism on Linux.
- Tessellation tolerances and printability thresholds (with the owner's sign-off on the values).
- The plant-model framework for the simulator.
- The check registry's code layout, and how rounds are recorded in the bundle.
