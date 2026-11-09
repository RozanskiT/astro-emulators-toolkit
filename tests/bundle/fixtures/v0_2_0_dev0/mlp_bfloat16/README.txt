Astro Emulators Toolkit Bundle

Summary:
  model: mlp
  release: unreleased
  bundle_format_version: 1
  config_schema_version: 1
  spec_version: 1
  weights_layout: params_plus_model_state_v1
  model_family_id: mlp_v1
  fingerprint_evaluation: absent
  task: regression
  fit_method: gradient
  solver_params: not provided
  solver_diagnostics: not provided
  solver_design_matrix: not provided
  role_paths: {'input_leaf': 'inputs/parameters', 'output_leaf': 'outputs/predictions'}

Domain:
  input_domain: not provided
  reference_scaling_inputs: not provided
  reference_scaling_outputs: not provided
  extras: not provided

Provenance:
  toolkit_version: 0.2.0.dev0
  created_at: 2026-11-09T06:39:12.829815+00:00
  python_version: 3.12.13
  working_tree_git_commit: 65feb0b19c26d7925b61c2ffd33ecd3d02603c88

spec:
spec_version: 1
inputs:
  structure_tree:
    parameters: None
  channel_names_tree: None
  leaf_units_tree: None
  channel_units_tree: None
  leaf_meanings_tree: None
  channel_meanings_tree: None
outputs:
  structure_tree:
    predictions: None
  channel_names_tree: None
  leaf_units_tree: None
  channel_units_tree: None
  leaf_meanings_tree: None
  channel_meanings_tree: None

Note: this bundle is the canonical emulator artifact. Physical-space composition is external.
