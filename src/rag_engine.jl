# ============================================================================
# rag_engine.jl — RAGAssistant
# ============================================================================
# Retrieval-Augmented Generation pipeline:
#   1. Accept user query
#   2. Encode to vector → search Qdrant → gather Top-K context
#   3. Build a prompt combining context + query
#   4. Send to LLM API (DeepSeek or Claude) via HTTP
#   5. Return grounded response
# ============================================================================

module RAGAssistant

using HTTP
using JSON3
using LinearAlgebra: norm

# Bring sibling modules into scope (they live in parent module MiniVDB)
using ..MultimodalEncoder: encode_text
using ..DBClient: search, SearchResult

export RAGConfig, retrieve, generate, ask, build_prompt

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

"""
    RAGConfig(; provider, api_key, model, system_prompt, temperature, max_tokens)

Configuration for the LLM backend used by the RAG pipeline.

Fields:
- `provider`      — `:deepseek` or `:claude`
- `api_key`       — API key (read from env var `DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY` if empty)
- `model`         — model ID string
- `system_prompt` — system-level instruction prepended to each request
- `temperature`   — sampling temperature (0.0–2.0)
- `max_tokens`    — cap on generated tokens
"""
struct RAGConfig
    provider::Symbol
    api_key::String
    model::String
    system_prompt::String
    temperature::Float64
    max_tokens::Int
end

"""
Default constructor: reads API key from environment if not provided.
"""
function RAGConfig(;
        provider::Symbol      = :deepseek,
        api_key::String       = "",
        model::String         = provider == :deepseek ? "deepseek-chat" : "claude-sonnet-4-6-20250514",
        system_prompt::String = "You are a helpful, precise AI assistant. Answer questions based " *
                                "on the provided context. If the context does not contain relevant " *
                                "information, say so honestly.",
        temperature::Float64  = 0.3,
        max_tokens::Int       = 1024,
    )

    key = api_key
    if isempty(key)
        key = provider == :deepseek ? get(ENV, "DEEPSEEK_API_KEY", "") :
                                       get(ENV, "ANTHROPIC_API_KEY", "")
    end
    if isempty(key)
        @warn "No API key provided for provider :$provider. " *
              "Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY env var, or pass api_key= explicitly."
    end

    return RAGConfig(provider, key, model, system_prompt, temperature, max_tokens)
end

# ============================================================================
# CONTEXT RETRIEVAL
# ============================================================================

"""
    retrieve(client, collection, encoder, query; k = 5)

Encode `query` with the multimodal encoder, search Qdrant, and return
a list of `SearchResult` objects containing the best-matching payloads.
"""
function retrieve(client, collection::String, encoder, query::String; k::Int = 5)
    qvec = encode_text(encoder, query)
    return search(client, collection, qvec; k = k)
end

# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

"""
    build_prompt(query::String, context::Vector{SearchResult}) -> String

Combine the user query with retrieved context into an LLM-ready prompt.

The returned string includes the retrieved documents/texts/images as a
numbered list for the LLM to reference.
"""
function build_prompt(query::String, context::Vector{<:Any})::String
    parts = String[
        "You are answering a question using retrieved context from a multimodal knowledge base.\n",
        "--- RETRIEVED CONTEXT ---\n",
    ]

    for (i, hit) in enumerate(context)
        push!(parts, "[$(i)] (score: $(round(hit.score; digits=4))) ")
        if haskey(hit.payload, "text")
            push!(parts, string(hit.payload["text"]))
        elseif haskey(hit.payload, "path")
            push!(parts, "[IMAGE: $(hit.payload["path"])] ")
            if haskey(hit.payload, "description")
                push!(parts, string(hit.payload["description"]))
            end
        end
        push!(parts, "\n")
    end

    push!(parts, "\n--- USER QUERY ---\n")
    push!(parts, query)
    push!(parts, "\n\nProvide a clear, grounded answer based *only* on the context above. ")
    push!(parts, "Cite the relevant context numbers [1], [2], etc. when you use them.")

    return join(parts)
end

# ============================================================================
# LLM GENERATION
# ============================================================================

"""
    generate(config::RAGConfig, prompt::String) -> String

Send the assembled prompt to the configured LLM API and return the
generated text.
"""
function generate(config::RAGConfig, prompt::String)::String
    if config.provider == :deepseek
        return _call_deepseek(config, prompt)
    elseif config.provider == :claude
        return _call_claude(config, prompt)
    else
        error("Unsupported provider: :$(config.provider). Use :deepseek or :claude.")
    end
end

# ---- DeepSeek API ----------------------------------------------------------

function _call_deepseek(config::RAGConfig, prompt::String)::String
    body = JSON3.write(Dict(
        "model"       => config.model,
        "temperature" => config.temperature,
        "max_tokens"  => config.max_tokens,
        "messages"    => [
            Dict("role" => "system", "content" => config.system_prompt),
            Dict("role" => "user",   "content" => prompt),
        ],
    ))

    headers = Dict(
        "Content-Type"  => "application/json",
        "Authorization" => "Bearer $(config.api_key)",
    )

    resp = HTTP.post(
        "https://api.deepseek.com/v1/chat/completions";
        body    = body,
        headers = headers,
    )

    if resp.status != 200
        error("DeepSeek API error: HTTP $(resp.status) — $(String(resp.body))")
    end

    data = JSON3.read(resp.body)
    return string(data["choices"][1]["message"]["content"])
end

# ---- Claude API ------------------------------------------------------------

function _call_claude(config::RAGConfig, prompt::String)::String
    body = JSON3.write(Dict(
        "model"       => config.model,
        "max_tokens"  => config.max_tokens,
        "temperature" => config.temperature,
        "system"      => config.system_prompt,
        "messages"    => [
            Dict("role" => "user", "content" => prompt),
        ],
    ))

    headers = Dict(
        "Content-Type"      => "application/json",
        "x-api-key"         => config.api_key,
        "anthropic-version" => "2023-06-01",
    )

    resp = HTTP.post(
        "https://api.anthropic.com/v1/messages";
        body    = body,
        headers = headers,
    )

    if resp.status != 200
        error("Claude API error: HTTP $(resp.status) — $(String(resp.body))")
    end

    data = JSON3.read(resp.body)
    return string(data["content"][1]["text"])
end

# ============================================================================
# END-TO-END PIPELINE (convenience)
# ============================================================================

"""
    ask(config::RAGConfig, client, collection::String, encoder, query::String;
        k::Int = 5) -> String

End-to-end RAG pipeline: retrieve → build prompt → generate → return answer.

This is the single function you call for a complete RAG query.
"""
function ask(config::RAGConfig, client, collection::String, encoder, query::String;
             k::Int = 5)::String

    @info "RAG query: \"$(query)\" (top-$k)"

    # Step 1: retrieve
    results = retrieve(client, collection, encoder, query; k = k)

    if isempty(results)
        return "No relevant context found in the knowledge base for: \"$query\""
    end

    @info "Retrieved $(length(results)) context(s)."

    # Step 2: build prompt
    prompt = build_prompt(query, results)

    # Step 3: generate
    @info "Calling $(config.provider) LLM (model: $(config.model))..."
    answer = generate(config, prompt)

    return answer
end

end # module RAGAssistant
