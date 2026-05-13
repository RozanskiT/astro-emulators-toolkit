Astro Emulators Toolkit Bundle

Summary:
  model: mlp
  release: payne-flux-reference-example@0.1.0 (released)
  bundle_format_version: 1
  config_schema_version: 1
  spec_version: 1
  weights_layout: params_plus_model_state_v1
  model_family_id: mlp_v1
  fingerprint_evaluation: present
  task: regression
  fit_method: gradient
  solver_params: not provided
  solver_diagnostics: not provided
  solver_design_matrix: not provided
  role_paths: {'input_leaf': 'inputs/parameters', 'output_leaf': 'outputs/flux'}

Domain:
  input_domain: {'kind': 'box_v1', 'max_tree': {'parameters': [7000.0, 5.0, 0.30000001192092896]}, 'min_tree': {'parameters': [4500.0, 2.5, -0.30000001192092896]}, 'storage': {'filename': 'input_domain.safetensors', 'format': 'safetensors_v1', 'layout': 'split_minmax_tree_v1'}, 'value_space': 'physical_input_dict_tree_v1'}
  reference_scaling_inputs: {'applies_to': 'inputs', 'kind': 'affine_minmax_v1', 'max_tree': {'parameters': [7000.0, 5.0, 0.30000001192092896]}, 'min_tree': {'parameters': [4500.0, 2.5, -0.30000001192092896]}, 'source_space': 'physical_input_dict_tree_v1', 'storage': {'filename': 'reference_scaling_inputs.safetensors', 'format': 'safetensors_v1', 'layout': 'split_minmax_tree_v1'}, 'target_space': 'canonical_input_dict_tree_v1'}
  reference_scaling_outputs: {'applies_to': 'outputs', 'kind': 'affine_minmax_v1', 'max_tree': {'flux': 1.0}, 'min_tree': {'flux': 0.0}, 'source_space': 'canonical_output_dict_tree_v1', 'storage': {'filename': 'reference_scaling_outputs.safetensors', 'format': 'safetensors_v1', 'layout': 'split_minmax_tree_v1'}, 'target_space': 'physical_output_dict_tree_v1'}
  extras: ['notes', 'wavelength_angstrom']

Provenance:
  toolkit_version: 0.1.0
  created_at: 2026-04-20T04:22:43.298214+00:00
  python_version: 3.12.13
  git_commit: 8d46f0a70583bee4bfca376c90ab10ffc35ab9c3

spec:
input_domain:
  kind: box_v1
  max_tree:
    parameters:
      - 7000.0
      - 5.0
      - 0.30000001192092896
  min_tree:
    parameters:
      - 4500.0
      - 2.5
      - -0.30000001192092896
  storage:
    filename: input_domain.safetensors
    format: safetensors_v1
    layout: split_minmax_tree_v1
  value_space: physical_input_dict_tree_v1
inputs:
  channel_meanings_tree:
    parameters:
      - effective temperature
      - surface gravity
      - metallicity [Fe/H]
  channel_names_tree:
    parameters:
      - teff
      - logg
      - feh
  channel_units_tree:
    parameters:
      - K
      - dex
      - dex
  leaf_meanings_tree:
    parameters: stellar labels
  leaf_units_tree: None
  structure_tree:
    parameters: None
outputs:
  channel_meanings_tree: None
  channel_names_tree: None
  channel_units_tree: None
  leaf_meanings_tree:
    flux: continuum-normalized flux vector on the shared wavelength grid
  leaf_units_tree:
    flux: dimensionless
  structure_tree:
    flux: None
reference_scaling_inputs:
  applies_to: inputs
  kind: affine_minmax_v1
  max_tree:
    parameters:
      - 7000.0
      - 5.0
      - 0.30000001192092896
  min_tree:
    parameters:
      - 4500.0
      - 2.5
      - -0.30000001192092896
  source_space: physical_input_dict_tree_v1
  storage:
    filename: reference_scaling_inputs.safetensors
    format: safetensors_v1
    layout: split_minmax_tree_v1
  target_space: canonical_input_dict_tree_v1
reference_scaling_outputs:
  applies_to: outputs
  kind: affine_minmax_v1
  max_tree:
    flux: 1.0
  min_tree:
    flux: 0.0
  source_space: canonical_output_dict_tree_v1
  storage:
    filename: reference_scaling_outputs.safetensors
    format: safetensors_v1
    layout: split_minmax_tree_v1
  target_space: physical_output_dict_tree_v1
spec_version: 1

Note: this bundle is the canonical emulator artifact. Physical-space composition is external.
