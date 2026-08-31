# Evaluating retrieval

## Golden sets
A golden set is a list of questions paired with the chunk or document that should
be retrieved. Twenty questions is enough to detect a regression. Without one, every
tuning decision is a guess dressed up as intuition.

## Hit rate and MRR
Hit at k is the fraction of questions where the gold chunk appears in the top k
results. Mean reciprocal rank averages one divided by the rank of the first correct
result, so it rewards putting the right answer first rather than merely somewhere.

## Faithfulness
Faithfulness measures whether the generated answer is supported by the retrieved
passages. A system that retrieves well and then hallucinates is still broken. The
cheapest check is whether the model refused when the answer was genuinely absent.
