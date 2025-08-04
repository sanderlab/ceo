# Combinatorial Entropy Optimization

A C++ implementation of the [combinatorial entropy optimization (CEO)](https://doi.org/10.1186/gb-2007-8-11-r232) algorithm. Originial publication: Reva, B., Antipin, Y. & Sander, C. Determinants of protein function revealed by combinatorial entropy optimization. *Genome Biol* **8**, R232 (2007).

## Usage

Follow the instructions in `src/README.txt` to compile the C++ code and place the binary in your `$PATH`.

To cluster sequences in a multiple sequence alignment (MSA), run the Python script:

`python ceo.py [options] msa_file`

`msa_file`: MSA in FASTA format

Options:

`--max-col-gap`: Maximum fraction of gaps allowed in an MSA column. Default: 0.3. Columns with more gaps are not used for clustering.

`-A`: Range of the *A* parameter (`begin,end,step`). Default: 0.5,0.975,0.025. 

`--threads`: Number of threads to use. Default: 4.
