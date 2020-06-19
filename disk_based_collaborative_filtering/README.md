# Disk Based Collaborative Filtering

- Project 2 of Big Data subject. Part B
- Implemented by:
   - Theodoros Mandilaras
   - cs2.190018
- MSc DIT/EKPA 2019-2020

---

## How to run?
The program is made to run on python3 only!

The program gets most of the inputs as arguments by using the *Argparse* module. Only the *user_id* input is given in a 
while true loop. In each loop you can search for a different user, with the same method.
 
### Output
In the end of the processing the program prints 20 rows with the results. Each row contains the movieId, 
the similarity score, and the method that used to calculate it ('u' for user, 'i' for item). The results are sorted, 
staring with the best suggestion 

### Quit
The programs terminates by giving the 'q' letter for input or with the Ctrl+C shortcut.

### Arguments
The required arguments are:
- `-m` or `--method`: It gets which method to use in order to predict. Values can be only: "user", "item" or "mix"
- `-r` or `--ratings_path`: The path where the ratings file is located

Optional arguments are:
- `--load` or `-l`: If this flag exists it will try to load the pivot tables from the known path

You can check the help menu with the `-h` argument.

---

### Executable
The main executable program is called `collab_filtering.py` and it can be executed as:

`python collab_filtering.py -m <method> -r path/to/ratings.csv` 

or if you want to load the pivot tables (see bellow) in order to avoid recreating them, execute like:

`python collab_filtering.py -m <method> -r path/to/ratings.csv --load`

---

## Multiple Pivot Tables
The program, in order to predict and recommend movies, it parse the ratings file line by line the csv file and creates 
multiple pivot tables which they are stored in the **`pivot-tables` folder which is created in the working directory**.
Each table contains the ratings for 30.000 users. The creation of those tables requires about 10 minutes  

Those tables are **csr sparse matrices**. They are stored with the `save_npz` and loaded with the `load_npz`.

The tables can be reused by using the `--load` argument. 

---

# Examples

## user-based
`python disk_based_collaborative_filtering/collab_filtering.py -m user -r ml-25m/ratings.csv`

## item-based
`python disk_based_collaborative_filtering/collab_filtering.py -m item -r ml-25m/ratings.csv  --load`

## mix
`python disk_based_collaborative_filtering/collab_filtering.py -m mix -r ml-25m/ratings.csv` 