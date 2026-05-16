# ============================================================================
# encoder.jl — MultimodalEncoder
# ============================================================================
# Encodes text AND images into a shared 512-d CLIP embedding space.
#
# Backends:
#   1. CLIP via PythonCall.jl (real — requires `pip install clip torch Pillow`)
#   2. Mock via deterministic hash projection (fallback — zero Python deps)
# ============================================================================

module MultimodalEncoder

using LinearAlgebra: norm
using Random: MersenneTwister, randn

export CLIPEncoder, encode_text, encode_image, mock_text_embed

# ============================================================================
# TYPE DEFINITION
# ============================================================================

"""
    CLIPEncoder

Holds configuration for a CLIP-based multimodal encoder.

Constructors
------------
- `CLIPEncoder()`                              → mock backend, 512-d
- `CLIPEncoder(; backend=:mock, dim=512)`      → mock backend, configurable dim
- `CLIPEncoder(model_name::String)`            → real CLIP via PythonCall
  (e.g. `CLIPEncoder("ViT-B/32")`)

Fields
------
- `model_name` — HuggingFace / OpenAI CLIP variant
- `dim`        — embedding dimension (512 for ViT-B, 768 for ViT-L)
- `backend`    — `:clip` (PythonCall) or `:mock` (deterministic hash)
- `_loaded`    — internal flag; set true once Python model is loaded
"""
mutable struct CLIPEncoder
    model_name::String
    dim::Int
    backend::Symbol
    _loaded::Bool
    _py_clip::Any
    _py_model::Any
    _py_preprocess::Any

    # ---- inner constructor: all fields explicit (dispatches via new) ---------
    function CLIPEncoder(model_name::String, dim::Int, backend::Symbol,
                         loaded::Bool, py_clip, py_model, py_preprocess)
        return new(model_name, dim, backend, loaded, py_clip, py_model, py_preprocess)
    end
end

# ---- outer constructor: named CLIP model (real Python backend) ---------------
"""
    CLIPEncoder(model_name::String)

Create an encoder backed by a real CLIP model loaded via PythonCall.

Examples
--------
    CLIPEncoder("ViT-B/32")
    CLIPEncoder("ViT-L/14")
"""
function CLIPEncoder(model_name::String)
    dims = Dict{String, Int}(
        "ViT-B/32" => 512,
        "ViT-B/16" => 512,
        "ViT-L/14" => 768,
    )
    dim = get(dims, model_name, 512)
    return CLIPEncoder(model_name, dim, :clip, false, nothing, nothing, nothing)
end

# ---- outer constructor: mock backend (keyword-only, zero-dependency) ---------
"""
    CLIPEncoder(; backend::Symbol = :mock, dim::Int = 512)

Create an encoder with a deterministic word-hash projection backend.

This is the **safe default** — no Python, no network, always works.
Call `CLIPEncoder()` to get a 512-dimensional mock encoder.

Examples
--------
    CLIPEncoder()                        # mock, 512-d
    CLIPEncoder(backend = :mock, dim = 128)
"""
function CLIPEncoder(; backend::Symbol = :mock, dim::Int = 512)
    return CLIPEncoder("mock-$(dim)d", dim, backend, true, nothing, nothing, nothing)
end

# ============================================================================
# REAL CLIP BACKEND  (PythonCall)
# ============================================================================

function _ensure_clip_loaded!(enc::CLIPEncoder)
    enc._loaded && return nothing

    try
        PythonCall.pyexec("""
import torch
import clip
from PIL import Image
        """)
        clip = PythonCall.pyimport("clip")
        torch = PythonCall.pyimport("torch")
        model, preprocess = clip.load(enc.model_name; device = "cpu")
        model.eval()
        enc._py_clip = clip
        enc._py_model = model
        enc._py_preprocess = preprocess
        enc._loaded = true
        @info "CLIP model '$(enc.model_name)' loaded successfully (backend: PythonCall)."
    catch e
        @warn "Failed to load CLIP via PythonCall. Install with:\n" *
              "  pip install torch clip Pillow\n" *
              "  Falling back to mock embeddings." exception = e
        enc.backend = :mock
        enc._loaded = true
    end
    return nothing
end

"""
    encode_text(enc::CLIPEncoder, text::String) -> Vector{Float32}

Produce a normalised text embedding vector.

Dispatch: calls the real CLIP text encoder if available, otherwise falls
back to `mock_text_embed`.
"""
function encode_text(enc::CLIPEncoder, text::String)::Vector{Float32}
    if enc.backend == :clip
        _ensure_clip_loaded!(enc)
        # Re-check after load attempt — may have fallen back
        if enc.backend == :clip
            return _clip_encode_text(enc, text)
        end
    end
    return mock_text_embed(text; dim = enc.dim)
end

function _clip_encode_text(enc::CLIPEncoder, text::String)::Vector{Float32}
    tokens = enc._py_clip.tokenize([text])
    torch  = PythonCall.pyimport("torch")
    with(torch.no_grad) do
        feats = enc._py_model.encode_text(tokens)  # shape (1, 512)
        arr   = feats[0, :].cpu().numpy()
        vec   = PythonCall.pyconvert(Vector{Float32}, arr)
    end
    nrm = norm(vec)
    return iszero(nrm) ? vec : vec ./ Float32(nrm)
end

"""
    encode_image(enc::CLIPEncoder, image_path::String) -> Vector{Float32}

Produce a normalised image embedding vector from a file on disk.

Dispatch: calls the real CLIP image encoder if available, otherwise falls
back to `mock_text_embed(image_path)` (a deterministic hash of the path).
"""
function encode_image(enc::CLIPEncoder, image_path::String)::Vector{Float32}
    if enc.backend == :clip
        _ensure_clip_loaded!(enc)
        if enc.backend == :clip
            return _clip_encode_image(enc, image_path)
        end
    end
    # Mock: treat the file path as the text to embed (hashes the path string)
    return mock_text_embed(image_path; dim = enc.dim)
end

function _clip_encode_image(enc::CLIPEncoder, image_path::String)::Vector{Float32}
    PIL   = PythonCall.pyimport("PIL.Image")
    torch = PythonCall.pyimport("torch")
    img   = PIL.open(image_path).convert("RGB")
    inp   = enc._py_preprocess(img).unsqueeze(0)
    with(torch.no_grad) do
        feats = enc._py_model.encode_image(inp)  # shape (1, 512)
        arr   = feats[0, :].cpu().numpy()
        vec   = PythonCall.pyconvert(Vector{Float32}, arr)
    end
    nrm = norm(vec)
    return iszero(nrm) ? vec : vec ./ Float32(nrm)
end

# ============================================================================
# MOCK BACKEND  (zero-dependency, deterministic)
# ============================================================================

const _MOCK_SEED_BASE = UInt64(2024)

"""
    mock_text_embed(text::String; dim::Int = 512) -> Vector{Float32}

Deterministic word-hash embedding for testing without external dependencies.

Same algorithm as the standalone `embed_mock`: tokenize, seed an RNG per
token with `hash(token)`, draw a random unit vector, sum across tokens,
normalize.  Returns `Float32` for compat with Qdrant.
"""
function mock_text_embed(text::String; dim::Int = 512)::Vector{Float32}
    tokens = split(lowercase(text))
    isempty(tokens) && return zeros(Float32, dim)

    acc = zeros(Float64, dim)  # accumulate in Float64 for precision
    for token in tokens
        seed = _MOCK_SEED_BASE ⊻ hash(token)
        rng  = MersenneTwister(seed)
        v    = randn(rng, Float64, dim)
        v   ./= norm(v)
        acc .+= v
    end

    nrm = norm(acc)
    return iszero(nrm) ? zeros(Float32, dim) : Float32.(acc ./ nrm)
end

end # module MultimodalEncoder
