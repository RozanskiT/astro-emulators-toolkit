# Examples

Astronomer-facing usage starts in `basic/` and then moves to `advanced/`.
`development/` and `experimental/` are maintainer and research surfaces, not the primary onboarding path.

## Reading order
1. `basic/` (stable entry point)
2. `advanced/` (integration and internals)
3. `development/` (maintainer benchmarks, bundle builders, and longer-run recipes)
4. `experimental/` (prototype models/tasks)

## Dataset policy for astronomer-facing examples
- Astronomer-facing examples use data from `examples/examples_datasets/`.
- Randomized datasets (`irregular_*`) are used for train/validation splits in current stable training examples.
- Regular grids (`regular_*`) are reserved for dedicated evaluation/fitting demos.
- `isochrones/` and `tracks/` are both preserved in-repo.
- README-listed basic and advanced scripts use canonical dict input/output trees only.
