## Overview

The following are results obtained on the Ikat benchmark.

Notes:

- Basline means: BM25 + T5 + MonoT5 + BART.
- Baseline + x rounds of simulation means that feedback is collected after one loop of the baseline pipeline. The feedback is then used as input to the pipeline.

| Method | Run File Path | Components | Recall | MAP | MRR | NDCG | NDCG @ 3 | 
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Baseline   | [run file](../data/generated_conversations/bm25_t5_monot5_bart/bm25_t5_monot5_bart.run) | T5 + BM25 + MonoT5 + BART (1K pass.)  | 0.1754   | 0.0491   | 0.3189  | 0.1447  | 0.1546  |
| Baseline + 1 round of simulation   | [run file](../data/generated_conversations/bm25_t5_t5_bart_llama2_1round/bm25_t5_t5_bart_llama2_1round.run) | T5 + BM25 + MonoT5 + BART (1K pass.) with 1 round of llama 2 simulated feedback  | 0.1346  | 0.0312   | 0.1953  | 0.0774  | 0.0842  |
| Baseline + 2 round of simulation   | [run file](../data/generated_conversations/bm25_t5_t5_bart_llama2_2round/bm25_t5_t5_bart_llama2_2round.run) | T5 + BM25 + MonoT5 + BART (1K pass.) with 2 round of llama 2 simulated feedback  | 0.1262 | 0.0307   | 0.1847  | 0.0922  | 0.0832 |
| FeedBack Ranking Baseline   | [run file](../data/generated_conversations/bm25_t5_bart_llama2_t5_bart/bm25_t5_bart_llama2_t5_bart.run) | T5 + BM25 + BART + Llama 2 feedback + MonoT5 + BART  | 0.1754 | 0.0258   | 0.1709  | 0.1074  | 0.0596 |
| FeedBack Ranking v2   | [run file](../data/generated_conversations/bm25_t5_monot5_bart_llama2_t5_monot5_bart/bm25_t5_monot5_bart_llama2_t5_monot5_bart.run) | T5 + BM25 + BART + Llama 2 feedback + T5 + MonoT5 + BART  | 0.1754 | 0.0372 | 0.2335  | 0.1241  | 0.1096 |
| Colbert   | [run file](../data/generated_conversations/colbert_bart/colbert_bart.run) | Colber + BART  | 0.0790 | 0.0191 | 0.1878  | 0.0603  | 0.0736 |
| Baseline + 1 round of simulation (prompted)   | [run file](../data/generated_conversations/prompted_simulation_baseline/prompted_simulation_baseline.run) | T5 + BM25 + MonoT5 + BART (1K pass.) with 1 round of prompted llama 2 simulated feedback  | 0.0975 | 0.0157 | 0.1394  | 0.0676  | 0.0628 |
| Single Action Agent   | [run file](../data/generated_conversations/single_action_agent/single_action_agent.run) | T5 + BM25 + MonoT5 + BART (1K pass.) with single action agent  | 0.1543 | 0.0501 | 0.2600  | 0.1280  | 0.1293 |
| Manual Baseline   | [run file](../data/generated_conversations/manual_baseline/manual_baseline.run) | BM25 + MonoT5 + BART (1K pass.) on manual queries | 0.3208 | 0.1039 | 0.5129  | 0.2645  | 0.2726 |
| Manual Baseline w/ PTKBS   | [run file](../data/generated_conversations/manual_baseline_with_ptkbs/manual_baseline_with_ptkbs.run) | BM25 + MonoT5 + BART (1K pass.) on manual queries with ptkbs and default utterance | 0.2862 | 0.0921 | 0.5041  | 0.2411  | 0.2348 |
| Manual Baseline w/ PTKBS (no default)   | [run file](../data/generated_conversations/manual_baseline_with_ptkbs_no_default/manual_baseline_with_ptkbs_no_default.run) | BM25 + MonoT5 + BART (1K pass.) on manual queries with ptkbs and no default utterance | 0.3206 | 0.1029 | 0.5427  | 0.2634  | 0.2830 |
| Multi Action Agent (3 rounds of recursion)   | [run file](../data/generated_conversations/multi_action_agent/multi_action_agent.run) | Multi Action Agent with a default clarification pipeline | 0.1558 | 0.0470 | 0.2679  | 0.1278  | 0.1379 |
| Baseline + 1 Round of simulation (RM3 rewriter)   | [run file](../data/generated_conversations/tuned_simulation_baseline_With_rm3_rewriter/tuned_simulation_baseline_with_rm3_rewriter.run) | RM3 (5 terms) + BM25 + MonoT5 + BART (1K pass.) with 1 round of llama 2 simulated feedback | 0.1347 | 0.0247 | 0.1498  | 0.0932  | 0.0659 |
| Single Action Pipeline Agent with GPT3.5 Feedback Rewriter (No PTKBs)   | [run file](../data/generated_conversations/single_action_agent_pipeline_with_feedback_rewriter/single_action_agent_pipeline_with_feedback_rewriter.run) | Single Action Agent with GPT3.5 rewriter | 0.1797 | 0.0543 | 0.2921  | 0.1439  | 0.1431 |
| Single Action Pipeline Agent with GPT3.5 Feedback Rewriter (With IN and PTKBs)   | [run file](../data/generated_conversations/single_action_agent_pipeline_with_feedback_rewriter_v2/single_action_agent_pipeline_with_feedback_rewriter_v2.run) | Single Action Agent with GPT3.5 rewriter | 0.2763 | 0.0827 | 0.4149  | 0.2139  | 0.2175 |
| Baseline with GPT3 Rewriter   | [run file](../data/generated_conversations/standard_baseline_gpt3_rewriter/standard_baseline_gpt3_rewriter.run) | GPT3 + BM25 + MonoT5 + BART (1K pass.) | 0.2152 | 0.0648 | 0.3615  | 0.1755  | 0.1787 |
| Baseline with GPT3 Simulator  | [run file](../data/generated_conversations/gpt3_simulation_baseline/gpt3_simulation_baseline.run) | T5 + BM25 + MonoT5 + BART with GPT3 simulation | 0.2179 | 0.0466 | 0.3151  | 0.1490  | 0.1279 |
| Single Action Llama Agent Pipeline with GPT3 Simulator  | [run file](../data/generated_conversations/single_action_agent_with_gpt3/single_action_agent_with_gpt3.run) | Single Action Agent with GPT3 simulator | 0.2090 | 0.0437 | 0.3148  | 0.1483  | 0.1362 |
| GPT3 rewriter and simulator  | [run file](../data/generated_conversations/openai_rewriter_and_simulator/openai_rewriter_and_simulator.run) | GPT3 Rewriter and Simulator | 0.2551 | 0.0729 | 0.2003  | 0.1949  | 0.2034 |