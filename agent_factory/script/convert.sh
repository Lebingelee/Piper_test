INPUT_DIR="data/merged/dual_merged.h5" 
OUTPUT_DIR=

python3 -m agent_factory.script.convert_to_flattened \
    -i "$INPUT_DIR" -o "$OUTPUT_DIR"