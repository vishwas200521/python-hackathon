import subprocess
import os

# List of Python files to run
files_to_run = ['kmrl.py','new.py']

# Directory where the files are located (optional, if they are not in the same folder)
scripts_dir = ''

for file in files_to_run:
    # Construct the full path to the script
    script_path = os.path.join(scripts_dir, file)

    print(f"Running{file} ...")
    
    # Run the script using the Python interpreter
    # The 'check=True' argument will raise an error if the script fails
    try:
        subprocess.run(['python', script_path], check=True)
        print(f"{file} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}\n")