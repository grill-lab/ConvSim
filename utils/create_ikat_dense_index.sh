input_directory="../../data/datasets/ikat/raw_docs"
output_parent_directory="../../data/indexes/dense/ikat"
encoder_model="castorini/tct_colbert-v2-hnp-msmarco"
num_gpus=4

# Function to process a single file
process_file() {
    file="$1"
    gpu_index="$2"
    gpu_index=$(((gpu_index-1) % num_gpus))
    filename=$(basename -- "$file")
    extension="${filename##*.}"
    filename="${filename%.*}"

    output_directory="${output_parent_directory}/${filename}"
    mkdir -p "${output_directory}"

    echo "Output directory: ${output_directory}"
    echo "Encoder model: ${encoder_model}"
    echo "GPU index: ${gpu_index}"

    # Create a separate lock file for each GPU
    lock_file="/tmp/gpu_lock_${gpu_index}"

    flock "${lock_file}" python -m pyserini.encode \
        input --corpus "${file}" --fields text \
        output --embeddings "${output_directory}" --to-faiss \
        encoder --encoder "${encoder_model}" --fields text --batch 32 --fp16 --device "cuda:${gpu_index}"

    echo "Processing ${filename} completed on GPU ${gpu_index}."
}

# Export the function to make it available to parallel
export -f process_file
export input_directory output_parent_directory encoder_model lock_file num_gpus

# Use parallel to process files concurrently on multiple GPUs
find "${input_directory}" -name '*.jsonl' | \
    parallel -j8 --eta process_file {} '{#}'
