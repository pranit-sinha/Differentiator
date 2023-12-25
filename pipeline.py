import argparse
import os
import subprocess
from multiprocessing import pool
from OpenSlideGenerator.OpenSlide import getTiles
from OpenSlideGenerator.csv_generator import generate_csv
from decisionTree import run_decision_tree

subprocess.run(["bash", "-c", r"""echo -e "\e[1;36m
________  .__  _____  _____                            __  .__        __                
\______ \ |__|/ ____\/ ____\___________   ____   _____/  |_|__|____ _/  |_  ___________ 
 |    |  \|  \   __\\   __\/ __ \_  __ \_/ __ \ /    \   __\  \__  \\   __\/  _ \_  __ \
 |    `   \  ||  |   |  | \  ___/|  | \/\  ___/|   |  \  | |  |/ __ \|  | (  <_> )  | \/
/_______  /__||__|   |__|  \___  >__|    \___  >___|  /__| |__(____  /__|  \____/|__|   
        \/                     \/            \/     \/             \/                   
\033[0m"""])

def generate_patches_and_csv(input_image, output_folder):
    getTiles(input_image, output_folder)
    generate_csv(output_folder)

def run_decision_tree_on_csv(csv_file):
    run_decision_tree(csv_file)

def perform_both_steps(input_image, output_folder, csv_file):
    generate_patches_and_csv(input_image, output_folder)

    run_decision_tree_on_csv(csv_file)

def main():
    parser = argparse.ArgumentParser(description="Command-line interface for image processing.")
    parser.add_argument("option", type=int, choices=[1, 2, 3], help="Choose option 1, 2, or 3. 1: Generate patches and CSV from a TIFF image. 2: Run a CSV through decisionTree.py. 3: Perform both steps together.")
    
    if parser.parse_args().option == 1:
        input_image = input("Enter the path to the TIFF image: ")
        output_folder = input("Enter the path to the output folder for patches: ")
        patch_size = int(input("Enter the patch size (e.g., 128): "))
        overlap = float(input("Enter the overlap percentage (e.g., 0.2 for 20% overlap): "))
        generate_patches_and_csv(input_image, output_folder)

    elif parser.parse_args().option == 2:
        csv_file = input("Enter the path to the CSV file: ")
        run_decision_tree_on_csv(csv_file)

    elif parser.parse_args().option == 3:
        input_image = input("Enter the path to the TIFF image: ")
        output_folder = input("Enter the path to the output folder for patches: ")
        csv_file = os.path.join(output_folder, "output.csv")  # Assuming default output file name
        perform_both_steps(input_image, output_folder, csv_file)

if __name__ == "__main__":
    main()
    pool = Pool(mp.cpu_count())
    sizes = pool.map(list_creator, folders)
    pool.close()
    pool.join()
