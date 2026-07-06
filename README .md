# RouterRL: Budget-Constrained LLM Routing with Reinforcement Learning

A reinforcement learning system that learns to route incoming queries between a **weak (cheap)** and **strong (expensive)** language model under a fixed global token budget, balancing response quality against the number of queries it can afford to answer.

## Problem

Given a stream of incoming queries and a shared, depleting token budget, decide — query by query — which model should answer it. Sending everything to the strong model maximizes per-query quality but exhausts the budget quickly, leaving later queries unanswered. Sending everything to the weak model preserves budget but sacrifices quality. The agent has to learn a pacing strategy that adapts to query difficulty and remaining budget.

## Requirements

```
gymnasium
stable-baselines3
datasets
pandas
numpy
scikit-learn
matplotlib
```

## Dataset

[`JiaqiXue/mmr-routing-20k`](https://huggingface.co/datasets/JiaqiXue/mmr-routing-20k) on Hugging Face — conversation-level queries paired with:
- `strong_score` / `weak_score`: quality (0–10) if answered by the strong vs. weak model
- Linguistic features per query: `difficulty`, `query_len_words`, `is_short_query`, `has_pronoun_reference`, `is_first_turn`, `turn_position_ratio`, `has_continuation_marker`, `word_overlap_prev_query`

Conversations are split 80/20 into train/test **by `conversation_hash`**, so all turns of a given conversation stay together in one split (no leakage across turns of the same conversation).

## Environment (`AdvancedRoutingEnv`)

A custom `gymnasium.Env` formulating routing as a sequential decision problem.

**State (9-dim, continuous):**
The 8 z-scored linguistic features (normalized using train-set mean/std, reused at test time to avoid leakage) plus `remaining_budget / initial_budget`.

**Action (discrete, 2):**
- `0` → route to the weak model
- `1` → route to the strong model

**Cost model:**
```
strong_cost = query_len + 80 + 2.0 * query_len
weak_cost   = query_len + 15 + 0.8 * query_len
```
Strong is both a higher fixed cost and a steeper per-word cost than weak.

**Reward:**
```
reward = model_quality_score - tax_rate * actual_cost      (tax_rate = 0.025)
```
- If the agent picks strong but can't afford it, the environment force-falls-back to weak and applies an extra penalty (`-2.0`).
- If the agent can't even afford weak, the episode terminates with a `-10.0` bankruptcy penalty.

**Budget:** fixed at 10,000 tokens per episode.

## Models Compared

| Strategy | Description |
|---|---|
| **Always Weak** | Naive baseline — every query to the weak model |
| **Always Strong** | Naive baseline — every query to the strong model |
| **Threshold Heuristic** | Rule-based: route to strong if `difficulty > 5.0` |
| **Random Forest** | Supervised classifier trained on oracle labels (`strong_net_reward > weak_net_reward`, computed from ground-truth scores) — a label-informed reference point rather than a pure heuristic |
| **RL Agent (PPO)** | `stable-baselines3` PPO, MLP policy (`[128, 128]`), trained for 250,000 timesteps on the training environment |

## Results

Final evaluation runs each strategy through `AdvancedRoutingEnv` on the held-out test set and reports:
- **Queries Answered** — how many queries were served before the budget ran out
- **Quality Score** — cumulative net reward (quality minus cost tax, minus any penalties) across the episode
- **Remaining Tokens** — unused budget at episode end

*(Run the notebook to populate this table with current numbers.)*

| Strategy | Queries Answered | Quality Score | Remaining Tokens |
|---|---|---|---|
| Always Weak | 116 | 362.53 | 1061.0 |
| Always Strong | 49 | 215.08 | 3.0 |
| Threshold Heuristic | 82 | 417.20 | 8.0 |
| Random Forest | 73 | 412.82 | 33.0 |
| **RL Agent** | **85** | **435.22** | **49.0** |

The RL agent achieves the highest quality score of all strategies, while answering more queries than either the threshold heuristic or the Random Forest baseline and leaving a modest token reserve unused. It beats "Always Weak" on quality despite answering far fewer queries, and dramatically outperforms "Always Strong" on both quality and coverage — evidence it has learned a genuine pacing strategy rather than defaulting to either extreme.

Note the caveat in "Known Limitations" below: this run reflects a single deterministic episode over one (leading) slice of the test set, so treat these numbers as a snapshot rather than a robust estimate — re-run over multiple randomized episodes for a confidence interval before drawing strong conclusions.

## Known Limitations / Next Steps

- **Episode length vs. budget mismatch**: with a 10,000-token budget and ~40–80 tokens/query, an episode only covers a few hundred queries before bankruptcy — far short of the full train/test set sizes. Currently every episode starts at row 0, so training and evaluation only ever see the same leading slice of data. Randomizing the start index (or shuffling per-episode) is needed to expose the agent to the full dataset and to get a statistically meaningful evaluation (mean ± std over multiple runs, not a single deterministic pass).

## How to Run

1. **Install dependencies** (see [Requirements](#requirements) below):
   ```bash
   pip install gymnasium stable-baselines3 datasets pandas numpy scikit-learn matplotlib
   ```

2. **Run the notebook top to bottom** (`Router_RL.ipynb`). It will:
   - Pull the dataset directly from Hugging Face (`JiaqiXue/mmr-routing-20k`) — no manual download needed, just an internet connection.
   - Build the `AdvancedRoutingEnv` environment and split conversations into train/test.
   - Train the Random Forest baseline on oracle labels.
   - Train the PPO agent for 250,000 timesteps (this is the slowest step — expect it to take a few minutes on CPU).
   - Run the final evaluation, printing the results table (Queries Answered / Quality Score / Remaining Tokens) for all five strategies.

3. **Check the output** — the final cells print the comparison table and any accompanying plots. No files are written to disk by default; everything runs and displays inline.

No GPU is required — PPO with an MLP policy on this state size trains fine on CPU.

## Learning Goals

- Formulating a real-world resource-allocation problem as an RL environment
- Designing states, actions, and rewards for a budget-constrained sequential decision task
- Understanding quality–cost trade-offs under scarcity
- Benchmarking a learned policy against heuristic and supervised baselines
