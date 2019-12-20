### LyceumBase

[![](https://github.com/Lyceum/LyceumBase.jl/workflows/CI/badge.svg)](https://github.com/Lyceum/LyceumBase.jl/actions)

This is where `AbstractEnv` is specified for the Lyceum framework. It provides a set of functions that can be future specialzed for different physics engines or other MDP-centric problem setups.

# Functions

```julia
  statespace(env::AbstractEnv)
  getstate(env::AbstractEnv)
  getstate!(state, env::AbstractEnv)

  observationspace(env::AbstractEnv)
  getobs(env::AbstractEnv)
  getobs!(observation, env::AbstractEnv)

  actionspace(env::AbstractEnv)
  getaction(env::AbstractEnv)
  getaction!(action, env::AbstractEnv)
  setaction!(env::AbstractEnv, action)

  getreward(env::AbstractEnv)

  evalspace(env::AbstractEnv)       # Task evaluation metric
  geteval(env::AbstractEnv)         # that can differ from reward

  reset!(env::AbstractEnv)          # Reset to a fixed, initial state.
  reset!(env::AbstractEnv, state)   # Reset to `state`
  randreset!(env::AbstractEnv)      # Reset to a random initial state.

  step!(env::AbstractEnv)           # Step the environment, return reward.
  step!(env::AbstractEnv, action)   # Apply `action`, step, return reward.
```

