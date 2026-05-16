#!/usr/bin/env julia
# ============================================================================
# add_missing.jl — Programmatically add missing stdlib / utility packages
#                   to the active project environment.
#
# Usage (REPL):
#   julia> include("add_missing.jl")
#
# Usage (terminal):
#   julia --project=. add_missing.jl
# ============================================================================

using Pkg

# ---- 1. Activate the project -----------------------------------------------
proj_dir = @__DIR__
Pkg.activate(proj_dir)
println("[1/3] Activated project at:  $proj_dir\n")

# ---- 2. Add missing packages -----------------------------------------------
# These stdlibs are bundled with Julia but must be listed in Project.toml
# for Julia ≥ 1.9 to resolve `using` statements inside local modules.
#
#   Random       → MersenneTwister, randn          (used by encoder.jl)
#   Statistics   → mean, std                        (common in vector DB ops)
#   Logging      → @info, @warn, @error             (logging infrastructure)
#   SparseArrays → sparse matrix ops                (optional, for scale)
# ---------------------------------------------------------------------------

_missing = [
    ("Random",       "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"),
    ("Statistics",   "10745b16-79ce-11e8-11f9-7d13ad32a3b2"),
    ("Logging",      "56ddb016-857b-54e1-b83d-db4d58db5568"),
]

println("[2/3] Adding missing packages …")
for (name, uuid) in _missing
    try
        Pkg.add(Pkg.PackageSpec(; name = name, uuid = UUID(uuid)))
        println("       ✓ $name  ($uuid)")
    catch e
        @warn "       ⚠ Could not add $name — it may already be present." exception = e
    end
end

# ---- 3. Resolve & instantiate ----------------------------------------------
println("\n[3/3] Resolving + instantiating …")
Pkg.resolve()
Pkg.instantiate()

println("""

    Done. The following stdlibs are now in Project.toml [deps]:

      Random       — required by  src/encoder.jl  (MersenneTwister, randn)
      Statistics   — available for future vector-stats operations
      Logging      — available for structured logging

    If you still see errors, run:

      julia --project=. fix_env.jl

    … to purge stale caches and regenerate Manifest.toml from scratch.
""")
