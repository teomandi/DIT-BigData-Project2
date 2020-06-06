# Disk Based Collaborative Filtering

- Project 2 of Big Data subject. Part B
- Implemented by:
   - Theodoros Mandilaras
   - cs2.190018
- DIT/EKPA 2019-2020

---

## How to run?
The program is made to run on python3 only!

The program gets most of the inputs as arguments by using the *Argparse* module. Only the *user_id* input is given in a 
while true loop. In each loop you can search for a different user.
 
### Output
In the end of the processing the program prints 20 rows with the results. Each row contains the movieId, 
the similarity rank, and the method that used to calculate it ('u' for user, 'i' for item). The results are sorted, 
staring with the best suggestion 

### Quit
The programs terminates by giving the 'q' letter for input or with the Ctrl+C shortcut.

### Arguments
The required arguments are:
- -m or --method: It gets which method to use in order to predict. Values can be only: "user", "item" or "mix"
- -r or --ratings_path: The path where the ratings file is located

Optional arguments are:
- -p or --pivot_table_path: The path to store/load the pivot table
- --load or -l: If this flag exists it will try to load the pivot table from the path
- --store or -s: If this flag exists it will try to store the pivot table in the given path

You can check the help menu with the -h argument.

---

### Executable
The main executable program is called `collab_filtering.py` and it can be executed as:

`python collab_filtering.py -m <method> -r path/to/ratings.csv` 

or if you want to store the pivot table (see bellow) in order to avoid recreating it, execute like:

`python collab_filtering.py -m <method> -r path/to/ratings.csv -p 'path/to/wanted/name.sparce --store`

or if it is already stored like:

`python collab_filtering.py -m <method> -r path/to/ratings.csv -p 'path/to/wanted/name.sparce --load`

---

## Pivot Table
The program, in order to predict and recommend movies, it generates a sparse pivot table. This table contains the 
information about all the rates a user made for every movie. This pivot table is created by parsing the ratings csv and
requires about **10 minutes** to be build with respect of the memory. 

### Reusing the pivot table
In order to avoid recreating the pivot table every time, it can be created once and then, to be reused. By using the
`--store` argument the program will store that sparse matrix in the path that was given from the `-p` argument. 
Similarly, if the table is already stored, it can be loaded using the `--load` argument form the given path. For loading
and storing the program uses the *pickle* module.

The pivot table needs about 10 minutes to be created.