# Development history, not final scores

These files retain the earlier residential validation experiments and
training-only source-review sensitivity analysis. Random Forest and histogram
boosting were exploratory library benchmarks. They are not the required own
implementations, not the final selected models, and not a claimed own-ensemble
bonus. Source-review records do not change the final primary training targets.

The final pipeline reproduces its complete recorded search with
`python -m src.run_all`. Earlier exploratory experiments can be reproduced
separately with `python -m src.improve_residential`, which writes its own outputs.

Full-data Stage 1 classification used a 210,000 AZN threshold. Final residential
classification uses 207,000 AZN. Do not compare the two as identical tasks.
