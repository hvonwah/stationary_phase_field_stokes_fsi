#!/bin/bash

mkdir -p output
echo "" > output/fsi_in_exact_ellipse.txt

python3 fsi_in_ellipse_stat.py -o 2 -hmax 0.12  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 2 -hmax 0.06  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 2 -hmax 0.03  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 2 -hmax 0.015 >> output/fsi_in_exact_ellipse.txt

python3 fsi_in_ellipse_stat.py -o 3 -hmax 0.12  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 3 -hmax 0.06  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 3 -hmax 0.03  >> output/fsi_in_exact_ellipse.txt

python3 fsi_in_ellipse_stat.py -o 4 -hmax 0.12  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 4 -hmax 0.06  >> output/fsi_in_exact_ellipse.txt
python3 fsi_in_ellipse_stat.py -o 4 -hmax 0.03  >> output/fsi_in_exact_ellipse.txt
