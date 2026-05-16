#!/usr/bin/env julia
# ============================================================================
# sample_kaggle.jl — Flickr8k Dataset Ingestion Pipeline
# ============================================================================
# Parses the Flickr8k Image Captioning dataset and ingests image embeddings
# + caption embeddings into the MiniVDB vector store (Qdrant or offline).
#
# Usage:
#   julia --project=. sample_kaggle.jl [max_images]
#
# Prerequisites:
#   - Dataset placed at  dataset/Flickr8k/
#   - Julia deps:        CSV, DataFrames  (handled by Project.toml)
#   - Qdrant running:    docker compose up -d   (optional — offline fallback)
# ============================================================================

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using CSV
using DataFrames

# ---- Load MiniVDB (local module) -------------------------------------------
# Use explicit include() so the module inherits the root environment's deps
# — avoids the Pkg.develop path-layout requirement (src/src/MiniVDB.jl).
include(joinpath(@__DIR__, "src", "MiniVDB.jl"))
using .MiniVDB

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset may live at the project root or inside src/
const _CANDIDATE_ROOTS = [
    joinpath(@__DIR__, "dataset", "Flickr8k"),
    joinpath(@__DIR__, "src", "dataset", "Flickr8k"),
]
const DATASET_ROOT  = findfirst(isdir, _CANDIDATE_ROOTS) !== nothing ?
                      _CANDIDATE_ROOTS[findfirst(isdir, _CANDIDATE_ROOTS)] :
                      _CANDIDATE_ROOTS[1]   # default (will fail with a clear error)
const IMAGES_DIR    = joinpath(DATASET_ROOT, "Images")
const CAPTIONS_FILE = joinpath(DATASET_ROOT, "captions.txt")

# ============================================================================
# INGESTION PIPELINE
# ============================================================================

"""
    ingest_flickr8k(max_images::Int = 50)

Parse the Flickr8k dataset and ingest up to `max_images` unique images
(along with their 5 captions each) into MiniVDB.

Steps:
1. Read `captions.txt` into a DataFrame.
2. Extract the first `max_images` unique image filenames.
3. For each image:
   a. Embed the image via CLIP (or mock) → upsert into Qdrant / offline store.
   b. Embed each of its 5 captions → upsert individually.
4. Log progress to the terminal.

Returns the number of images actually processed.
"""
function ingest_flickr8k(max_images::Int = 50)
    println()
    println("="^68)
    println("  Flickr8k Ingestion Pipeline")
    println("="^68)
    println()

    # ---- 0. Verify dataset exists -------------------------------------------
    if !isdir(DATASET_ROOT)
        error("""
            Dataset directory not found: $DATASET_ROOT
            Please download the Flickr8k dataset from Kaggle and place it at:
              $DATASET_ROOT/
                ├── Images/        (8091 .jpg files)
                └── captions.txt   (CSV: image, caption)
            """)
    end

    if !isfile(CAPTIONS_FILE)
        error("captions.txt not found at $CAPTIONS_FILE")
    end

    # ---- 1. Read captions ---------------------------------------------------
    println("[1/4] Reading captions from  $CAPTIONS_FILE  ...")
    df = CSV.read(CAPTIONS_FILE, DataFrame; header = true)
    println("       Loaded $(nrow(df)) rows  (cols: $(join(names(df), ", ")))\n")

    # Normalise column names (handle possible casing / whitespace differences)
    rename!(df, Symbol.(strip.(lowercase.(string.(names(df))))))

    cols = Symbol.(names(df))

    if :image ∉ cols || :caption ∉ cols
        error("Expected columns 'image' and 'caption', got: $(cols)")
    end

    # ---- 2. Initialise MiniVDB bridge (before accessing encoder state) -------
    println("[2/4] Initialising MiniVDB bridge  ...")
    MiniVDB.init(; backend = :mock, dim = 512)
    enc = MiniVDB._BRIDGE_ENCODER[]
    println("       Encoder: $(enc.backend)  (dim=$(enc.dim))")
    if MiniVDB._BRIDGE_CLIENT[] !== nothing
        println("       Qdrant:  connected  ($(MiniVDB._BRIDGE_CLIENT[].base_url))")
    else
        println("       Qdrant:  offline mode  (in-memory storage)")
    end
    println()

    # ---- 3. Select unique images --------------------------------------------
    println("[3/4] Extracting unique images  (limit: $max_images) ...")
    unique_images = unique(df[!, :image])
    n_total = length(unique_images)
    n_process = min(max_images, n_total)
    selected = unique_images[1:n_process]
    println("       Found $n_total unique images in dataset.")
    println("       Will process the first $n_process.\n")

    # ---- 4. Ingest image by image -------------------------------------------
    println("[4/4] Ingesting images + captions  ...")
    println("-"^68)

    n_ingested = 0
    n_skipped  = 0
    t_start    = time()

    for (idx, img_filename) in enumerate(selected)
        img_path = joinpath(IMAGES_DIR, img_filename)

        # --- defensive check ------------------------------------------------
        if !isfile(img_path)
            @warn "  ⚠️  Image file missing — skipping." path = img_path
            n_skipped += 1
            continue
        end

        # --- embed & insert image -------------------------------------------
        img_payload = Dict{String, Any}(
            "type"    => "image",
            "content" => img_path,
            "dataset" => "Flickr8k",
        )
        MiniVDB.insert_image(img_path; payload = img_payload)

        # --- embed & insert each of the 5 captions --------------------------
        captions_df = df[df[!, :image] .== img_filename, :]
        ncaps = nrow(captions_df)

        for row in eachrow(captions_df)
            caption = row[!, :caption]
            text_payload = Dict{String, Any}(
                "type"         => "text",
                "content"      => caption,
                "source_image" => img_path,
                "dataset"      => "Flickr8k",
            )
            MiniVDB.insert_text(caption; payload = text_payload)
        end

        n_ingested += 1

        # --- verbose progress -----------------------------------------------
        perc = round(Int, idx / n_process * 100)
        bar  = "█"^round(Int, perc / 5) * "░"^(20 - round(Int, perc / 5))
        println("  [$bar]  $perc%  ($idx/$n_process)")
        println("  ✅  Image #$(n_ingested):  $(img_filename)")
        println("       ├─ $(ncaps) captions embedded")
        println("       └─ $(img_path)")
    end

    elapsed = round(time() - t_start; digits = 1)
    println("-"^68)
    println("  🎉  Ingestion complete!")
    println("       Images ingested  : $n_ingested")
    println("       Images skipped   : $n_skipped  (missing files)")
    println("       Captions indexed : $(n_ingested * 5)  (5 per image)")
    println("       Time elapsed     : $(elapsed)s")
    println("="^68)
    println()

    return n_ingested
end

# ============================================================================
# ENTRY POINT
# ============================================================================
# Runs automatically when the file is executed directly OR included from the
# REPL via  include("sample_kaggle.jl").  Use  isinteractive  to decide
# whether to exit on bad arguments (script mode) or just warn (REPL mode).
# ============================================================================

max_images = 50
if length(ARGS) >= 1
    val = tryparse(Int, ARGS[1])
    if val === nothing
        msg = "Usage:  julia --project=. sample_kaggle.jl [max_images]"
        if !isinteractive()
            println(msg)
            exit(1)
        else
            @warn msg * "  (using default: 50)"
        end
    else
        max_images = val
    end
end

n = ingest_flickr8k(max_images)
println("Indexed $n images into MiniVDB. Query them via:")
println("  julia --project=. -e 'using MiniVDB; MiniVDB.main()'")
println("  streamlit run app.py")
