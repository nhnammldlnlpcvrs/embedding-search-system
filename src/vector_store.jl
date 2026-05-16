# ============================================================================
# vector_store.jl — DBClient
# ============================================================================
# Thin, type-stable wrapper around the Qdrant REST API.
# ============================================================================

module DBClient

using HTTP
using JSON3

export QdrantClient, Point, SearchResult
export create_collection!, delete_collection!, upsert!, search, collection_exists

# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

"""
    QdrantClient(base_url::String = "http://localhost:6333")

Handle to a running Qdrant instance.

Fields:
- `base_url` — REST API root, e.g. `"http://localhost:6333"`
"""
struct QdrantClient
    base_url::String
end

"""
    Point(id, vector::Vector{Float32}, payload::Dict{String, Any})

A single item to be indexed. `id` can be `Int` or `String`.
"""
struct Point
    id::Union{Int, String}
    vector::Vector{Float32}
    payload::Dict{String, Any}
end

"""
    SearchResult(id, score::Float32, payload::Dict{String, Any})

A single hit returned by `search`.
"""
struct SearchResult
    id::Union{Int, String}
    score::Float32
    payload::Dict{String, Any}
end

# ============================================================================
# COLLECTION MANAGEMENT
# ============================================================================

"""
    collection_exists(client::QdrantClient, name::String) -> Bool

Check whether a collection exists in Qdrant.
"""
function collection_exists(client::QdrantClient, name::String)::Bool
    url = "$(client.base_url)/collections/$(name)"
    resp = HTTP.get(url; status_exception = false)
    return resp.status == 200
end

"""
    create_collection!(client::QdrantClient, name::String, dim::Int;
                       distance::String = "Cosine") -> Bool

Create a Qdrant collection configured for cosine-similarity search.

Returns `true` on success, `false` if the collection already exists.
"""
function create_collection!(client::QdrantClient, name::String, dim::Int;
                            distance::String = "Cosine")::Bool
    if collection_exists(client, name)
        @warn "Collection '$name' already exists — skipping creation."
        return false
    end

    url = "$(client.base_url)/collections/$(name)"
    body = JSON3.write(Dict(
        "vectors" => Dict(
            "size"     => dim,
            "distance" => distance,
        ),
    ))

    resp = HTTP.put(url; body = body, headers = _json_headers())
    if resp.status == 200
        @info "Collection '$name' created (dim=$dim, distance=$distance)."
        return true
    else
        error("Failed to create collection '$name': HTTP $(resp.status) — $(String(resp.body))")
    end
end

"""
    delete_collection!(client::QdrantClient, name::String) -> Bool

Delete a Qdrant collection and all its vectors. Irreversible.
"""
function delete_collection!(client::QdrantClient, name::String)::Bool
    url  = "$(client.base_url)/collections/$(name)"
    resp = HTTP.delete(url; status_exception = false)
    return resp.status == 200
end

# ============================================================================
# VECTOR OPERATIONS
# ============================================================================

"""
    upsert!(client::QdrantClient, collection::String, points::Vector{Point})

Insert or update a batch of points in a Qdrant collection.

Each `Point` must carry an `id`, a `vector::Vector{Float32}`, and a
`payload::Dict` containing the original text / image reference.
"""
function upsert!(client::QdrantClient, collection::String,
                 points::Vector{Point})::Nothing

    isempty(points) && return nothing

    url = "$(client.base_url)/collections/$(collection)/points?wait=true"

    body = JSON3.write(Dict(
        "points" => [
            Dict(
                "id"      => p.id,
                "vector"  => p.vector,
                "payload" => p.payload,
            ) for p in points
        ],
    ))

    resp = HTTP.put(url; body = body, headers = _json_headers())

    if resp.status != 200
        error("Upsert failed: HTTP $(resp.status) — $(String(resp.body))")
    end

    @info "Upserted $(length(points)) point(s) into '$collection'."
    return nothing
end

"""
    search(client::QdrantClient, collection::String,
           query_vector::Vector{Float32}; k::Int = 5, score_threshold::Float32 = 0.0f0)
    -> Vector{SearchResult}

Retrieve the Top-K nearest neighbours by cosine similarity.

Returns results in descending order of similarity.
"""
function search(client::QdrantClient, collection::String,
                query_vector::Vector{Float32};
                k::Int = 5,
                score_threshold::Float32 = 0.0f0)::Vector{SearchResult}

    url = "$(client.base_url)/collections/$(collection)/points/search"

    body = JSON3.write(Dict(
        "vector"        => query_vector,
        "limit"         => k,
        "with_payload"  => true,
        "score_threshold" => score_threshold,
    ))

    resp = HTTP.post(url; body = body, headers = _json_headers())

    if resp.status != 200
        error("Search failed: HTTP $(resp.status) — $(String(resp.body))")
    end

    data = JSON3.read(resp.body)
    results = SearchResult[]

    for hit in data["result"]
        push!(results, SearchResult(
            hit["id"],
            Float32(hit["score"]),
            Dict{String, Any}(hit["payload"]),
        ))
    end

    return results
end

# ============================================================================
# INTERNAL HELPERS
# ============================================================================

function _json_headers()
    return Dict(
        "Content-Type" => "application/json",
        "Accept"       => "application/json",
    )
end

end # module DBClient
