# Triton Puzzles Lite

Check out [Triton-Puzzles-Lite](https://github.com/SiriusNEO/Triton-Puzzles-Lite) for source and how to run.
Short of it is 
```bash
# Run all puzzles. Stop at the first failed one
TRITON_INTERPRET=1 python3 puzzles.py -a
# Run on GPU
python3 puzzles.py -a
# Only run puzzle 1
TRITON_INTERPRET=1 python3 puzzles.py -p 1
# More arguments, refer to help
python3 puzzles.py -h
```