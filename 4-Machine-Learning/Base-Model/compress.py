import gzip
import shutil

# Compress the file
with open("best_model.pth", "rb") as f_in:
    with gzip.open("best_model_compressed.pth.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

