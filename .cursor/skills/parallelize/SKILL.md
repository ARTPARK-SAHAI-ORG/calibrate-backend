---
name: parallelize
description: Plan a task for parallel execution, then run parallel agents to edit independent files/sections at the same time. Use whenever I say '/parallelize', 'parallelize this', 'run agents in parallel', or ask to split work across agents. ALSO apply this by default to any multi-step or multi-file task even if I don't say the word — plan the split first, then run independent subtasks in parallel.
---

# Parallelize

Plan first, then run parallel agents for independent work. Never skip the plan.

## Step 1 — Plan the split (show this before doing anything)

Keep it short and scannable, product-manager language:

1. **Break the task into subtasks** — the smallest chunks that each produce a clear result.
2. **Mark dependencies** — for each subtask, note what it needs from another subtask. A subtask is _independent_ only if it needs nothing from any other and touches files no other subtask touches.
3. **Group into waves:**
   - **Parallel wave** — independent subtasks, run at the same time.
   - **Sequential steps** — anything that depends on a prior result, run in order after.
4. **Assign file ownership** — each parallel agent owns a disjoint set of files (or clearly separated sections of one file). No two agents edit the same file at once. This is the rule that prevents collisions.

Show the plan as a simple list: each agent, what it does, which files it owns. Then proceed.

## Step 2 — Run the parallel wave

- Launch one agent per independent subtask, in parallel.
- Give each agent a tight scope: its task, its files, and an instruction to stay inside them.
- Don't launch anything that depends on an unfinished result.

## Step 3 — Run the sequential steps

After the parallel wave finishes, do the dependent work in dependency order. Some of these may themselves split into a new parallel wave — repeat Step 1 for them if so.

## Step 4 — Integrate and verify

- Pull the results together.
- Check the boundaries where agents' work meets (shared imports, function signatures, configs) — that's where parallel work breaks.
- Run the build/tests if available. Report what was done in a few lines.

## When NOT to parallelize

Skip it and just do the work sequentially when:

- The task is small or a single edit.
- Subtasks depend on each other in a chain (each needs the last one's output).
- The subtasks would all touch the same file.

Parallelism is for _independent_ work. Forcing it on dependent work creates merge conflicts and is slower, not faster. If nothing is independent, say so in one line and proceed sequentially.
