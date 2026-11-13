### seed_0/HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b

- evaluations 193, programs 22, selections 93, merge calls 0 (0 merges), final program 11; error lines 89 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_0/HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 149, programs 22, selections 64, merge calls 12 (8 merges), final program 19; error lines 146 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_0/hoverBench_HoverMultiHop_GEPA_qwen3-8b

- evaluations 221, programs 22, selections 101, merge calls 0 (0 merges), final program 13; error lines 10 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 2

1. **CC**: error 2026/11/13 07:00:13, iteration 5, evaluation in progress: val (started 2026/11/13 06:58:52, ended 2026/11/13 07:05:46); example val 110, claim=The race, where the  BRM P201 debuted at first place, was a Formula On; module create_query_hop2
   - in val of iteration 5: **CC** (program 1, val score 43.0; example 110 logged 0, other value 1); the list sampled at iteration 6 changes (entries: program 1 285 -> 286)
     - size (not part of the verdict): 96 selections and 0 merge calls after program 1; first changed selection at iteration 6: 1 of 550 list positions hold another program; program 1 val score 43.0, with the other value 43.33; final program 13 with 49.33
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 1 earlier evaluations with a later identical request, e.g. from base val of iteration 0: val of iteration 21; val of iteration 76
2. **CC**: error 2026/11/13 09:30:26, iteration 46, evaluation in progress: val (started 2026/11/13 09:27:14, ended 2026/11/13 09:39:42); example val 78, claim=The director of The Moth Diaries (born 1953) was a Canadian filmmaker ; module summarize1
   - in val of iteration 46: **CC** (program 10, val score 44.0; example 78 logged 0, other value 1); the list sampled at iteration 47 changes (entries: program 10 236 -> 237)
     - size (not part of the verdict): 55 selections and 0 merge calls after program 10; first changed selection at iteration 47: 1 of 2094 list positions hold another program; program 10 val score 44.0, with the other value 44.33; final program 13 with 49.33
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 8 earlier evaluations with a later identical request, e.g. from val of iteration 41: val of iteration 50; val of iteration 61 (and 4 more)

### seed_0/hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 261, programs 22, selections 116, merge calls 14 (12 merges), final program 13; error lines 3 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 1

1. **CC**: error 2026/11/13 07:38:11, iteration 16, evaluation in progress: val (started 2026/11/13 07:36:41, ended 2026/11/13 07:42:51); example val 104, claim=One of the composer of Bee (song) wrote works that were featured in co; module create_query_hop2
   - in val of iteration 16: **CC** (program 5, val score 40.67; example 104 logged 0, other value 1); program 5 takes part in the merge call at iteration 17
     - size (not part of the verdict): 101 selections and 11 merge calls after program 5; first changed selection at iteration 18: 25 of 1641 list positions hold another program; program 5 is in the merges of iteration 17 merged 1 and 5 via 0; program 5 val score 40.67, with the other value 41.0; final program 13 with 49.33
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 2 earlier evaluations with a later identical request, e.g. from val of iteration 2: val of iteration 56; val of iteration 107

### seed_1/HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b

- evaluations 220, programs 22, selections 109, merge calls 0 (0 merges), final program 6; error lines 73 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_1/HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 165, programs 22, selections 73, merge calls 13 (10 merges), final program 9; error lines 88 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_1/hoverBench_HoverMultiHop_GEPA_qwen3-8b

- evaluations 239, programs 22, selections 114, merge calls 0 (0 merges), final program 5; error lines 51 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 3

1. **CC**: error 2026/11/13 09:21:24, iteration 45, evaluation in progress: val (started 2026/11/13 09:19:32, ended 2026/11/13 09:29:36); example val 78, claim=The place where Steven J. Kachelmeier got his Chemistry Ph.D. and Univ; module summarize2
   - in val of iteration 45: **CC** (program 9, val score 40.0; example 78 logged 0, other value 1); the list sampled at iteration 46 changes (entries: program 0 229 -> 228, program 2 239 -> 238, program 4 230 -> 229, program 5 258 -> 257)
     - size (not part of the verdict): 69 selections and 0 merge calls after program 9; first changed selection at iteration 46: 26 of 1659 list positions hold another program; program 9 val score 40.0, with the other value 40.33; final program 5 with 50.33
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 3 earlier evaluations with a later identical request, e.g. from val of iteration 36: val of iteration 84
2. **CC**: error 2026/11/13 09:21:24, iteration 45, evaluation in progress: val (started 2026/11/13 09:19:32, ended 2026/11/13 09:29:36); example val 49, claim=A 1950's film that had a prequel called Davy Crockett and the River Pi; module create_query_hop3
   - in val of iteration 45: **CC** (program 9, val score 40.0; example 49 logged 0, other value 1); the list sampled at iteration 46 changes (entries: program 9 227 -> 228)
     - size (not part of the verdict): 69 selections and 0 merge calls after program 9; first changed selection at iteration 46: 2 of 1660 list positions hold another program; program 9 val score 40.0, with the other value 40.33; final program 5 with 50.33
   - left behind by an earlier val evaluation (extra, not in the rule): **NC**; no earlier evaluation of this split has a later identical request
3. **NC**: error 2026/11/13 13:52:48, iteration 114, evaluation in progress: val (started 2026/11/13 13:48:25, ended 2026/11/13 13:54:02); example val 237, claim=The actress who starred in The Legend of the Blue Sea, rose to fame af; module create_query_hop3
   - in val of iteration 114: **NC** (program 21, val score 44.0; example 237 logged 0, other value 1); decision unchanged and no later identical request
     - size (not part of the verdict): 0 selections and 0 merge calls after program 21; program 21 val score 44.0, with the other value 44.33; final program 5 with 50.33
   - left behind by an earlier val evaluation (extra, not in the rule): **NC**; no earlier evaluation of this split has a later identical request

### seed_1/hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 185, programs 22, selections 81, merge calls 12 (10 merges), final program 18; error lines 8 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 1

1. **CC**: error 2026/11/13 09:12:39, iteration 57, evaluation in progress: val (started 2026/11/13 09:11:07, ended 2026/11/13 09:18:47); example val 91, claim=The satellite mission of the European Space Astronomy Centre, was oper; module create_query_hop2
   - in val of iteration 57: **CC** (program 9, val score 45.67; example 91 logged 0, other value 1); the list sampled at iteration 58 changes (entries: program 0 219 -> 218, program 1 235 -> 234, program 3 234 -> 233, program 5 243 -> 242)
     - size (not part of the verdict): 28 selections and 6 merge calls after program 9; first changed selection at iteration 58: 26 of 1658 list positions hold another program; program 9 val score 45.67, with the other value 46.0; final program 18 with 56.67
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 5 earlier evaluations with a later identical request, e.g. from merge subsample of iteration 57: val of iteration 57

### seed_2/HotpotQABench_HotpotMultiHop_GEPA_qwen3-8b

- evaluations 331, programs 21, selections 167, merge calls 0 (0 merges), final program 14; error lines 60 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_2/HotpotQABench_HotpotMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 170, programs 22, selections 74, merge calls 11 (9 merges), final program 12; error lines 53 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 1

1. **CC**: error 2026/11/13 08:18:58, iteration 52, evaluation in progress: val (started 2026/11/13 08:10:06, ended 2026/11/13 08:27:14); example val 273, id=5ae657fe55429908198fa5a5; module final_answer
   - in val of iteration 52: **CC** (program 10, val score 61.67; example 273 logged 0, other value 1); program 10 takes part in the merge call at iteration 53
     - size (not part of the verdict): 26 selections and 5 merge calls after program 10; first changed selection at iteration 54: 5 of 1563 list positions hold another program; program 10 is in the merges of iteration 62 merged 10 and 12 via 2; program 10 val score 61.67, with the other value 62.0; final program 12 with 65.0
   - left behind by an earlier val evaluation (extra, not in the rule): **CC**; 2 earlier evaluations with a later identical request, e.g. from val of iteration 51: merge subsample of iteration 53; val of iteration 53

### seed_2/hoverBench_HoverMultiHop_GEPA_qwen3-8b

- evaluations 254, programs 22, selections 120, merge calls 0 (0 merges), final program 16; error lines 2 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 0


### seed_2/hoverBench_HoverMultiHop_GEPA-MERGE_qwen3-8b

- evaluations 221, programs 22, selections 97, merge calls 9 (7 merges), final program 21; error lines 12 (0 after the last evaluation)
- check of the rebuild against the log: all match
- lost requests: 2

1. **CC**: error 2026/11/13 07:28:21, iteration 11, evaluation in progress: val (started 2026/11/13 07:17:55, ended 2026/11/13 07:29:01); example val 267, claim=The creator of Yesterday Girl and Sidney Salkow are not of the same na; module create_query_hop3
   - in val of iteration 11: **CC** (program 2, val score 43.67; example 267 logged 0, other value 1); program 2 takes part in the merge call at iteration 12
     - size (not part of the verdict): 86 selections and 8 merge calls after program 2; first changed selection at iteration 12: 5 of 767 list positions hold another program; program 2 is in the merges of iteration 25 merged 2 and 3 via 0; iteration 42 merged 2 and 6 via 0; iteration 54 merged 2 and 11 via 0; program 2 val score 43.67, with the other value 44.0; final program 21 with 52.0
   - left behind by an earlier val evaluation (extra, not in the rule): **NC**; no earlier evaluation of this split has a later identical request
2. **CC**: error 2026/11/13 09:34:20, iteration 53, evaluation in progress: val (started 2026/11/13 09:31:49, ended 2026/11/13 09:40:02); example val 111, claim=The Colbert Report is a satirical Mary Sprague Miller Institute for Ba; module summarize2
   - in val of iteration 53: **CC** (program 11, val score 44.67; example 111 logged 0, other value 1); program 11 takes part in the merge call at iteration 54
     - size (not part of the verdict): 48 selections and 3 merge calls after program 11; first changed selection at iteration 55: 266 of 2022 list positions hold another program; program 11 is in the merges of iteration 54 merged 2 and 11 via 0; program 11 val score 44.67, with the other value 45.0; final program 21 with 52.0
   - left behind by an earlier val evaluation (extra, not in the rule): **NC**; no earlier evaluation of this split has a later identical request

