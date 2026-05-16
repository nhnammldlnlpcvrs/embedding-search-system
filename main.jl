#!/usr/bin/env julia
# ============================================================================
# main.jl — Multimodal RAG System Demo Entry Point
# ============================================================================
# Prerequisites:
#   1. Start Qdrant:     docker compose up -d
#   2. Install Julia deps: julia -e 'using Pkg; Pkg.instantiate()'
#   3. (Optional) Install CLIP:  pip install torch clip Pillow
#   4. (Optional) Set LLM key:   export DEEPSEEK_API_KEY="sk-..."
#   5. Run:               julia --project=. main.jl
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# If the module loads fail on first run, Pkg.instantiate() above will
# install the missing packages so a second run will succeed.
try
    using MiniVDB
catch e
    @warn "MiniVDB failed to load — retrying after dependency resolution." exception = (e)
    Pkg.resolve()
    using MiniVDB
end

MiniVDB.main()
