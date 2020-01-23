### LyceumBase

[![](https://github.com/Lyceum/LyceumBase.jl/workflows/CI/badge.svg)](https://github.com/Lyceum/LyceumBase.jl/actions)

LyceumBase presents a Markov Decision Process (MDP)-like abstraction API to facilitate the development of algorithms for research in trajectory optimization and control. The state, action, observation, and reward abstractions give a common interface that can be instantiated by making `Environments` that inherit the `AbstractEnvironment` type, alongside algorithms that use the interface.

For example, [LyceumMuJoCo](https://github.com/Lyceum/LyceumMuJoCo.jl) instantiates the LyceumBase functions building on the MuJoCo simulator for continuous control problems, which can be used by [LyceumAI](https://github.com/Lyceum/LyceumAI.jl) algorithms. Note that LyceumAI is agnostic to the underlying simulator as it uses the LyceumBase interface.

For exact details of the API please [see the docs](https://docs.lyceum.ml/dev/lyceumbase/abstractenvironment/).
