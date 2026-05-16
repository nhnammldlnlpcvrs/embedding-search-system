#!/usr/bin/env julia
# ============================================================================
# fix_env.jl — Repair a corrupted Julia project environment.
#
# Usage:
#   julia --project=. fix_env.jl
#     or, from the REPL:
#   julia> include("fix_env.jl")
# ============================================================================

using Pkg

# ---- 1. Activate the project -----------------------------------------------
proj_dir = @__DIR__
Pkg.activate(proj_dir)
println("[1/5] Activated project at:  $proj_dir")

# ---- 2. Delete stale Manifest.toml -----------------------------------------
manifest = joinpath(proj_dir, "Manifest.toml")
if isfile(manifest)
    rm(manifest; force = true)
    println("[2/5] Deleted stale Manifest.toml")
else
    println("[2/5] No Manifest.toml found — nothing to delete")
end

# ---- 3. Purge package-specific precompile caches (optional but thorough) ---
# Julia 1.12+ stores compiled caches under .julia/compiled/.
# Purging the depot of the problematic packages forces a clean re-download
# and re-precompile on the next `instantiate`.
println("[3/5] Purging stale caches for the target packages …")
_targets = ["CSV", "DataFrames", "HTTP", "JSON3", "PythonCall"]
for pkg in _targets
    try
        # `Pkg.gc` garbage-collects unused artifacts.
        # We explicitly `rm` the package from the depot to force a clean slate.
        depot = joinpath(DEPOT_PATH[1], "packages", pkg)
        if isdir(depot)
            rm(depot; force = true, recursive = true)
            println("       ✓ Purged depot cache for $pkg")
        else
            println("       - $pkg not in depot (already clean)")
        end
    catch e
        @warn "       ⚠ Could not purge $pkg — continuing …" exception = e
    end
end

# ---- 4. Resolve and instantiate --------------------------------------------
println("[4/5] Resolving dependency graph …")
Pkg.resolve()
println("       ✓ Resolution complete.")

println("       Instantiating (downloading + precompiling) …")
Pkg.instantiate()
println("       ✓ Instantiation complete.")

# ---- 5. Smoke-test: load every dependency ----------------------------------
println("[5/5] Smoke-test — loading all packages …")
_packages = [
    ("CSV",          "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"),
    ("DataFrames",   "a93c6f00-e57d-5684-b7b6-da5c2934e242"),
    ("HTTP",         "cd3eb016-35fb-5094-929b-558a96fad6f3"),
    ("JSON3",        "0f8b85d8-7281-11e9-16c2-39a750bddbf1"),
    ("PythonCall",   "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"),
]

_all_ok = true
for (name, uuid) in _packages
    try
        @eval using $Symbol(name)
        println("       ✓ $name  loaded successfully")
    catch e
        _all_ok = false
        @error "       ✗ $name  FAILED to load" exception = e
    end
end

println()
if _all_ok
    println("="^56)
    println("  ✅ Environment repaired successfully!")
    println("  All $(length(_packages)) packages loaded without errors.")
    println("="^56)
else
    println("="^56)
    println("  ⚠️  Some packages failed to load.")
    println("  Check the error messages above and verify:")
    println("    - Your Julia depot is healthy:  ", DEPOT_PATH[1])
    println("    - Network access to the General registry is available")
    println("    - No stale overrides in LocalPreferences.toml")
    println("="^56)
end
