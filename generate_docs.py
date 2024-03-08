import subprocess
import os 
import shutil

def update_docs():
    command = "pdoc --html lib/biosignal_toolbox --output-dir docs/"
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":

    docs_folder = "docs"
    if(os.path.isdir(docs_folder)): 
        shutil.rmtree(docs_folder)
    
    update_docs()
