# ============================================================================
# MiniVDB.jl — Multimodal RAG System (package entry point)
# ============================================================================
# Usage:
#   julia> ] activate .
#   julia> using MiniVDB
#   julia> MiniVDB.main()    # runs the demo pipeline
# ============================================================================

module MiniVDB

using HTTP
using LinearAlgebra: norm

# ---- Submodules (loaded in dependency order) -------------------------------
include("encoder.jl")        # MultimodalEncoder
include("vector_store.jl")   # DBClient
include("rag_engine.jl")     # RAGAssistant

# ---- Re-export public API --------------------------------------------------
using .MultimodalEncoder
using .DBClient
using .RAGAssistant

# Encoder
export CLIPEncoder, encode_text, encode_image, mock_text_embed

# DB
export QdrantClient, Point, SearchResult
export create_collection!, delete_collection!, upsert!, search, collection_exists

# RAG
export RAGConfig, ask, build_prompt, retrieve, generate

# Bridge API (called from Streamlit via juliacall)
export init, insert_text, insert_image, query_rag

# ============================================================================
# DEMO PIPELINE
# ============================================================================

"""
    main()

End-to-end multimodal RAG demonstration:

1. Connect to Qdrant (assumes docker-compose is running).
2. Index 2 images + 2 text documents via CLIP (or mock) encoder.
3. Run a multimodal text query: "a photo of an animal"
4. Generate a grounded LLM response using retrieved context.
"""
function main()
    println("="^72)
    println("  MiniVDB 2.0 — Multimodal RAG Pipeline")
    println("="^72)

    # ---- 1. Setup ----------------------------------------------------------
    println("\n[1/5] Initializing encoder...")
    # Use :mock backend by default (zero external deps).
    # Switch to CLIPEncoder("ViT-B/32") if you have Python + torch + clip installed.
    encoder = CLIPEncoder(; backend = :mock, dim = 512)
    @info "Encoder ready: backend=$(encoder.backend), dim=$(encoder.dim)"

    # ---- 2. Connect to Qdrant ----------------------------------------------
    println("\n[2/5] Connecting to Qdrant...")
    client = try
        c = QdrantClient("http://localhost:6333")
        # Health check
        HTTP.get("$(c.base_url)/healthz"; status_exception = true)
        @info "Qdrant is healthy at $(c.base_url)"
        c
    catch e
        @warn "Qdrant is not reachable — running in offline / mock mode." exception = e
        @info "Start Qdrant with:  docker compose up -d"
        nothing
    end

    collection_name = "multimodal_demo"

    # ---- 3. Index multimodal content ---------------------------------------
    println("\n[3/5] Indexing documents (2 texts + 2 images)...")

    # Create collection (512-d vectors)
    if client !== nothing
        delete_collection!(client, collection_name)  # clean slate
        create_collection!(client, collection_name, encoder.dim)
    end

    # --- Text documents ---
    texts = [
        "A golden retriever puppy plays fetch in a sunlit meadow full of wildflowers.",
        "Quantum computing leverages superposition and entanglement to solve " *
        "certain problems exponentially faster than classical computers.",
    ]

    # --- Image documents (paths; these do not need to exist for mock mode) ---
    images = [
        (path = "./data/cat_on_couch.jpg",
         desc = "An orange tabby cat sleeping peacefully on a velvet couch."),
        (path = "./data/eiffel_tower.jpg",
         desc = "The Eiffel Tower illuminated at night against a starry sky."),
    ]

    points = Point[]

    # Embed and stage text documents
    for (i, t) in enumerate(texts)
        vec = encode_text(encoder, t)
        push!(points, Point(
            i,
            vec,
            Dict{String, Any}("text" => t, "type" => "text"),
        ))
        println("   ✓ Text document #$i encoded  (|v|=$(round(norm(vec); digits=2)))")
    end

    # Embed and stage image documents
    for (j, img) in enumerate(images)
        # encode_image falls back to hashing the path for :mock backend
        vec = encode_image(encoder, img.path)
        payload = Dict{String, Any}(
            "type"        => "image",
            "path"        => img.path,
            "description" => img.desc,
        )
        push!(points, Point(
            length(texts) + j,
            vec,
            payload,
        ))
        println("   ✓ Image document #$(length(texts) + j) encoded  (|v|=$(round(norm(vec); digits=2)))")
    end

    # Upsert to Qdrant (or just hold in memory if offline)
    if client !== nothing
        upsert!(client, collection_name, points)
    end
    println("   → $(length(points)) multimodal items indexed.\n")

    # ---- 4. Multimodal search ----------------------------------------------
    println("[4/5] Multimodal query: \"a photo of an animal\"")
    query = "a photo of an animal"
    qvec  = encode_text(encoder, query)

    if client !== nothing
        results = search(client, collection_name, qvec; k = 4)
    else
        # Offline mode: compute cosine similarity in-process
        results = _offline_search(points, qvec; k = 4)
    end

    for (rank, r) in enumerate(results)
        @printf "   %d. [score=%.4f]  %s\n" rank r.score _summarize_payload(r.payload)
    end

    # ---- 5. RAG generation -------------------------------------------------
    println("\n[5/5] RAG generation via LLM...")

    config = RAGConfig(;
        provider  = :deepseek,
        api_key   = get(ENV, "DEEPSEEK_API_KEY", ""),
    )

    if !isempty(config.api_key)
        answer = ask(config, client, collection_name, encoder, query; k = 3)
        println("\n--- LLM Response ---")
        println(answer)
        println("--- end ---")
    else
        # Show the prompt that *would* be sent
        prompt = build_prompt(query, results)
        println("\n   (No API key set — showing the prompt that would be sent to the LLM:)\n")
        println("─────────────────────────────────────────────────────────────")
        println(prompt)
        println("─────────────────────────────────────────────────────────────")
        println("\n   Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY to enable live generation.")
    end

    println("\n" * "="^72)
    println("  Pipeline complete.")
    println("="^72)
end

# ---- Offline search fallback (when Qdrant is unreachable) ------------------

function _offline_search(points::Vector{Point}, qvec::Vector{Float32};
                         k::Int = 5)::Vector{SearchResult}
    db_results = SearchResult[]
    for p in points
        sim = Float32(sum(qvec .* p.vector) / (norm(qvec) * norm(p.vector)))
        push!(db_results, SearchResult(p.id, sim, p.payload))
    end
    sort!(db_results; by = r -> -r.score)
    return db_results[1:min(k, end)]
end

function _summarize_payload(payload::Dict{String, Any})::String
    t = get(payload, "type", "unknown")
    if t == "text"
        txt = get(payload, "text", "")
        return length(txt) > 70 ? txt[1:67] * "..." : txt
    elseif t == "image"
        return "[IMAGE] $(get(payload, "path", "?")) — $(get(payload, "description", ""))"
    end
    return string(payload)
end

# ============================================================================
# BRIDGE API — persistent state for Streamlit / juliacall
# ============================================================================

const _BRIDGE_ENCODER  = Ref{CLIPEncoder}()
const _BRIDGE_CLIENT   = Ref{Union{QdrantClient, Nothing}}(nothing)
const _BRIDGE_COLL      = Ref{String}("multimodal_rag")
const _BRIDGE_POINTS    = Point[]           # offline store when Qdrant is down
const _BRIDGE_NEXT_ID   = Ref{Int}(1)
const _BRIDGE_READY      = Ref{Bool}(false)

"""
    init(; backend::Symbol = :mock, dim::Int = 512) -> Bool

Initialise the persistent bridge state. Idempotent — safe to call
multiple times (subsequent calls are a no-op).

Returns `true` on success.
"""
function init(; backend::Symbol = :mock, dim::Int = 512)::Bool
    if _BRIDGE_READY[]
        return true
    end

    _BRIDGE_ENCODER[] = CLIPEncoder(; backend = backend, dim = dim)
    _BRIDGE_COLL[]    = "multimodal_rag"

    _BRIDGE_CLIENT[] = try
        c = QdrantClient("http://localhost:6333")
        HTTP.get("$(c.base_url)/healthz"; status_exception = true)
        delete_collection!(c, _BRIDGE_COLL[])
        create_collection!(c, _BRIDGE_COLL[], dim)
        @info "Bridge: Qdrant connected ($(c.base_url))."
        c
    catch e
        @warn "Bridge: Qdrant not available — running offline." exception = e
        nothing
    end

    empty!(_BRIDGE_POINTS)
    _BRIDGE_NEXT_ID[] = 1
    _BRIDGE_READY[]   = true
    @info "Bridge initialised (backend=$backend, dim=$dim)."
    return true
end

"""
    insert_text(text::String; payload::Dict{String, Any} = Dict{String, Any}()) -> Bool

Encode `text` and upsert it into the vector store.

If `payload` is provided it is merged on top of the auto-generated
metadata, so callers can attach extra fields (e.g. `source_image`).

Returns `true` on success.  In offline mode the point is held in-memory
so subsequent `query_rag` calls will still find it.
"""
function insert_text(text::String;
                     payload::Dict{String, Any} = Dict{String, Any}())::Bool
    isempty(strip(text)) && return false
    init()  # ensure ready

    enc = _BRIDGE_ENCODER[]
    vec = encode_text(enc, text)
    id  = _BRIDGE_NEXT_ID[]
    _BRIDGE_NEXT_ID[] = id + 1

    defaults = Dict{String, Any}("text" => text, "type" => "text")
    merged   = isempty(payload) ? defaults : merge(defaults, payload)
    point    = Point(id, vec, merged)

    if _BRIDGE_CLIENT[] !== nothing
        upsert!(_BRIDGE_CLIENT[], _BRIDGE_COLL[], [point])
    else
        push!(_BRIDGE_POINTS, point)
    end
    @info "Bridge: inserted text document #$id."
    return true
end

"""
    insert_image(file_path::String; payload::Dict{String, Any} = Dict{String, Any}()) -> Bool

Encode an image (by file path) and upsert it into the vector store.

Use `encode_image` from the MultimodalEncoder module so CLIP is used
when available, otherwise the deterministic mock hash of the path.

If `payload` is provided it is merged on top of the auto-generated
metadata, so callers can attach extra fields.

Returns `true` on success.
"""
function insert_image(file_path::String;
                      payload::Dict{String, Any} = Dict{String, Any}())::Bool
    isempty(strip(file_path)) && return false
    init()

    enc = _BRIDGE_ENCODER[]
    vec = encode_image(enc, file_path)
    id  = _BRIDGE_NEXT_ID[]
    _BRIDGE_NEXT_ID[] = id + 1

    fname = split(replace(file_path, "\\" => "/"), "/")[end]
    defaults = Dict{String, Any}(
        "type"        => "image",
        "path"        => file_path,
        "filename"    => fname,
        "description" => "Image: $fname",
    )
    merged = isempty(payload) ? defaults : merge(defaults, payload)
    point  = Point(id, vec, merged)

    if _BRIDGE_CLIENT[] !== nothing
        upsert!(_BRIDGE_CLIENT[], _BRIDGE_COLL[], [point])
    else
        push!(_BRIDGE_POINTS, point)
    end
    @info "Bridge: inserted image document #$id (path=$file_path)."
    return true
end

"""
    query_rag(query_data::String, query_type::String, top_k::Int, api_key::String)
      -> Tuple{String, Vector{Dict{String, Any}}}

Full RAG pipeline:
1. Encode `query_data` as text or image.
2. Retrieve top_k contexts from the vector store.
3. If `api_key` is non-empty, call the real LLM; otherwise return a
   high-quality mock response grounded in the retrieved context.

Returns:
- `answer::String` — the LLM (or mock) response.
- `contexts::Vector{Dict}` — retrieved hits, each containing
  `"type"`, `"content"`, and `"score"`.
"""
function query_rag(query_data::String, query_type::String, top_k::Int,
                   api_key::String)::Tuple{String, Vector{Dict{String, Any}}}
    init()

    # ---- encode query --------------------------------------------------------
    enc = _BRIDGE_ENCODER[]
    if query_type == "image"
        qvec = encode_image(enc, query_data)
    else
        qvec = encode_text(enc, query_data)
    end

    # ---- retrieve ------------------------------------------------------------
    if _BRIDGE_CLIENT[] !== nothing
        raw_results = search(_BRIDGE_CLIENT[], _BRIDGE_COLL[], qvec; k = top_k)
    else
        raw_results = _offline_search(_BRIDGE_POINTS, qvec; k = top_k)
    end

    # ---- marshal contexts for Python -----------------------------------------
    contexts = Dict{String, Any}[]
    for r in raw_results
        d = Dict{String, Any}(
            "type"  => get(r.payload, "type", "unknown"),
            "score" => Float64(r.score),
        )
        if get(r.payload, "type", "") == "text"
            d["content"] = get(r.payload, "text", get(r.payload, "content", ""))
            d["source_image"] = get(r.payload, "source_image", "")
        elseif get(r.payload, "type", "") == "image"
            d["content"] = get(r.payload, "path", get(r.payload, "content", ""))
            d["filename"] = get(r.payload, "filename", "")
            d["description"] = get(r.payload, "description", "")
        else
            d["content"] = string(r.payload)
        end
        push!(contexts, d)
    end

    # ---- generate answer -----------------------------------------------------
    if isempty(api_key) || isempty(contexts)
        answer = _mock_rag_response(query_data, contexts)
    else
        answer = try
            config = RAGConfig(; provider = :deepseek, api_key = api_key)
            ask(config, _BRIDGE_CLIENT[], _BRIDGE_COLL[], enc, query_data; k = top_k)
        catch e
            @warn "LLM call failed — returning mock response." exception = e
            _mock_rag_response(query_data, contexts)
        end
    end

    return answer, contexts
end

"""
    _mock_rag_response(query::String, contexts::Vector{Dict}) -> String

Generate a plausible, grounded mock response when no LLM API key is set.

The response references the retrieved contexts by index so the user can
see that the retrieval step worked correctly.
"""
function _mock_rag_response(query::String,
                            contexts::Vector{Dict{String, Any}})::String
    isempty(contexts) && return "No relevant documents found in the knowledge base for: \"$query\""

    lines = String[
        "## 🤖 Mock RAG Response\n\n",
        "*(This is a simulated response — set your `DEEPSEEK_API_KEY` for live LLM generation.)*\n\n",
        "Based on the retrieved context, here is a grounded answer to your query:\n\n",
        "> **Query:** \"$query\"\n\n",
        "---\n\n",
    ]

    for (i, ctx) in enumerate(contexts)
        score = round(get(ctx, "score", 0.0); digits = 4)
        tp    = get(ctx, "type", "unknown")
        if tp == "text"
            snippet = get(ctx, "content", "")[1:min(end, 100)]
            push!(lines, "**Source [$i]** (score: $score, type: text)\n")
            push!(lines, "> $snippet\n\n")
        elseif tp == "image"
            fname = get(ctx, "filename", "unknown")
            desc  = get(ctx, "description", "")
            push!(lines, "**Source [$i]** (score: $score, type: image)\n")
            push!(lines, "> 📷 $fname — $desc\n\n")
        end
    end

    push!(lines, "---\n\n")
    push!(lines, "**Synthesized Answer:** The top-ranked context ")
    if length(contexts) == 1
        push!(lines, "is highly relevant (score: $(round(contexts[1]["score"]; digits=4))) ")
        push!(lines, "and directly addresses the query. ")
    else
        push!(lines, "contains $(length(contexts)) items, with the best match ")
        push!(lines, "scoring $(round(contexts[1]["score"]; digits=4)). ")
        push!(lines, "Together, these sources provide a comprehensive picture related ")
        push!(lines, "to your query. ")
    end
    push!(lines, "In a production system, the LLM would use these retrieved chunks ")
    push!(lines, "to generate a detailed, cited, and grounded answer.")

    return join(lines)
end

end # module MiniVDB
