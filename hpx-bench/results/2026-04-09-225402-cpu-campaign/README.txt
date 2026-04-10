This directory contains:
- meta/run_info.txt            : machine/build metadata
- correctness/base.txt         : raw base correctness run
- correctness/hpx.txt          : raw HPX correctness run
- correctness/diff.txt         : normalized text diff
- bench/*.log                  : raw llama-bench outputs
- bench/index.tsv              : run index

Suggested manual review:
1. correctness/diff.txt should be empty or explainable
2. bench/*.log should show CPU-only behavior
3. compare throughput by case and thread count
