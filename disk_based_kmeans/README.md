# Disk Based KMeans

- Project 2 of Big Data subject. Part A
- Implemented by:
   - Theodoros Mandilaras
   - cs2.190018
- MSc DI/EKPA 2019-2020

---

## How to run?
The program is made to run on python3 only!

The program gets the inputs as arguments by using the *Argparse* module.
 
### Arguments
The required arguments are:
- `-k` or `--k` : The number of the wanted clusters
- `-p` or `--path`: The path for the  movie csv (for d1 only) or the **new** data file (will see bellow) 
- `-d` or `--distance`: Which distance function to use. Accepted values are: "d<1-4>"

Other optional arguments are:
- `-c` or `--chunk`: The chuck size of each step (default is  10000)
- `-t` or `--threshold`: The threshold of similarity which should be overcome in order a point to be accepted in a 
cluster (default is 0.4)
- `-r` or `--ratings_path`: The path for the ratings file (It is required if d3 od d4 is selected)
- `-e` or `--export`: If this argument is added, the program, will store the results in a CSV file in the current 
directory. 
Otherwise, it will print them

You can check the help menu with the -h argument.

---

### Executable
The main executable problem is called `kmeans.py` and it can by executed as:

`python kmeans.py -k 15 -p path/to/new_data_file.csv -d d2 --export` for example or 

`python kmeans.py -k 15 -p path/to/new_data_file.csv -d d4 -r path/to/ratings.csv -e`

*It is required the `-r` argument for the d3 and d4!*


--- 

## Output
When the program complete the computations, it will print for each cluster, its key, its clusteroid, and how many members
it has. If `d4` has been selected, it will print the genres, and the tags of the point which is set as clusteroid. For 
obvious reasons the ratings vector is not displayed.

Then, if the `--export` flag has been added, then the program will export the output  results in a csv file with name 
`<selected-distance>_results.csv`. In the csv file, the columns are movieId, Cluster Key. **Otherwise, if the argument 
is not mentioned, the program will print the results in the stdout!!**
 
---

## Create New Data File 

For the distances 2 and 4 in order to optimize the performance I create a new data_file which is **REQUIRED** for the 
distances 2 and 4.

This file is just like the movies csv file. The columns are [moviesId, tags, genres]. The moviesId and genres are the 
same as the original file. I create this file in order to add the tags in the same file. The tags are collected in each 
chunk and they are concatenated with a "|" as a separator, just like in genres. In this way, I can reuse the same 
functionality from d1 in d2. 

In order to create this new file run:

`python create_file.py --movie path/of/movies.csv --tags path/to/tags.csv --output path/to/store/new_file.csv`

or like the example below:

`python create_file.py -m ml-25m/movies.csv -t ml-25m/tags.csv -o ml-25m/new_data_file.csv`

**It takes about 10 seconds for the file to be created.**

---

# Examples

## Create New Data File
python disk_based_kmeans/create_file.py -t ml-25m/tags.csv -m ml-25m/movies.csv -o new_data_file.csv
## d1
python disk_based_kmeans/kmeans.py -k 15 -p new_data_file.csv -d d1 --export
## d2
python disk_based_kmeans/kmeans.py -k 15 -p new_data_file.csv -d d2 --export
## d3
python disk_based_kmeans/kmeans.py -k 15 -p new_data_file.csv -r ml-25m/ratings.csv -d d3 -t 0.2 -c 5000 --export
## d4
python disk_based_kmeans/kmeans.py -k 15 -p new_data_file.csv -r ml-25m/ratings.csv -d d4 -t 0.2 -c 5000 --export


