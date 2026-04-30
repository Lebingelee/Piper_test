import argparse
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agent_factory.data.converter import flatten_raw_h5

def main():
    parser = argparse.ArgumentParser(description="Realman Pipeline: Convert Merged Raw H5 to Flattened Training H5")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the merged raw .h5 file")
    parser.add_argument("-o", "--output", type=str, help="Output path (optional, defaults to input_flattened.h5)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        sys.exit(1)

    try:
        flatten_raw_h5(args.input, args.output)
    except Exception as e:
        print(f"Error during flattening: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
