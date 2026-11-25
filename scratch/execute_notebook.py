import json
import nbformat
from nbclient import NotebookClient

nb_path = "scratch/new_notebook.json"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook cells with nbclient...")
client.execute()

output_path = "VisuNav_Familiar_Person_Recognition.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Successfully executed and saved to {output_path}!")
