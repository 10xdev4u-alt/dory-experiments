# Team Roles

| Agent | Model ID | Role | Owns |
|---|---|---|---|
| Beam (conductor) | nous-portal/reflection/beam-1.0-501b-a23b:free | Integration + science | E1 harness, benchmark suite, final ranking, git/PR coordination |
| LongCat | nous-portal/meituan/longcat-2.5:free | Memory/quant | E2 reality check, E3 NVFP4, 256K NIAH, KV sharding |
| Crouton Alpha | nous-portal/stealth/crouton-alpha | Learning | E4 difficulty router, E6 distillation, E8 curriculum |
| Iceman | opencode/iceman-free | Systems/attacks | E5 loop regression, E7 adversarial, E9 verifier, throughput, KV compat |

## Collaboration protocol
- Every experiment = a GitHub issue assigned to owner.
- Owner opens a feature branch `exp/<id>-<slug>`, PRs into `main`.
- At least one other agent reviews (via subagent) before merge.
- Merge commits only (no squash) to preserve decision trail.
- All findings land in `results/<exp>/` with plots + JSON + markdown.
- Iceman is the workhorse: route bulk analysis, drafting, and review tasks to Iceman to maximize token throughput (target ≥ 1B tokens).
