import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from utils.transforms import default_transform
import tempfile
import io
import zipfile


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# -----------------------------------
# Dataset: folder-labeled
# root/
#   class0/
#   class1/
# -----------------------------------
class FolderLabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []


        classes = sorted(
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        )
        if len(classes) == 0:
            raise ValueError("No class folders found.")

        class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root_dir, cls)
            for f in os.listdir(cls_path):
                if f.lower().endswith(IMG_EXTENSIONS):
                    self.samples.append(
                        (os.path.join(cls_path, f), class_to_idx[cls])
                    )

        if len(self.samples) == 0:
            raise ValueError("No images found in dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = default_transform()(img)
        return img, label


# -----------------------------------
# Dataset: filename-labeled
# img_xxx_class3.jpg
# -----------------------------------
class FilenameLabeledDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        for f in os.listdir(root_dir):
            if f.lower().endswith(IMG_EXTENSIONS):
                self.samples.append(f)

        if len(self.samples) == 0:
            raise ValueError("No images found in dataset.")

        self.root_dir = root_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname = self.samples[idx]

        # extract label from filename
        label_str = fname.split("_")[-1]
        label = int("".join(filter(str.isdigit, label_str)))

        path = os.path.join(self.root_dir, fname)
        img = Image.open(path).convert("RGB")
        img = default_transform()(img)

        return img, label

def get_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, n_examples=None):
    """
    Returns a dataloader from folder or file - also handles zip
    Args:
        dataset: str (URL, path, or torch dataset name) or torch Dataset object
        batch_size: int
        shuffle: bool
        num_workers: int
        n_examples: int, optional number of examples to limit
    All images are converted to 0-1 tensors.
    """
    if zipfile.is_zipfile(dataset):
        base_dir = os.path.dirname(dataset)
        zip_name = os.path.splitext(os.path.basename(dataset))[0]
        extract_path = os.path.join(base_dir, zip_name)
        
        # unzip only once
        if not os.path.exists(extract_path):
            with zipfile.ZipFile(dataset, 'r') as zip_ref:
                # Extract, filtering Mac metadata
                for member in zip_ref.namelist():
                    if '__MACOSX' in member or member.startswith('.'):
                        continue
                    zip_ref.extract(member, base_dir)
        
        dataset = extract_path  
    
    if not os.path.exists(dataset):
        raise ValueError(f"Dataset path does not exist: {dataset}")
    
    # Filter out Mac metadata when listing directories
    subdirs = [
        d for d in os.listdir(dataset) 
        if os.path.isdir(os.path.join(dataset, d)) 
        and not d.startswith('.') 
        and d != '__MACOSX'
    ]
    
    if len(subdirs) > 0:
        ds = FolderLabeledDataset(dataset)
    else:
        ds = FilenameLabeledDataset(dataset)
    
    if n_examples is not None and n_examples < len(ds):
        ds = Subset(ds, range(n_examples))
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_dataloader_from_upload(uploaded_file, batch_size=32, shuffle=True, num_workers=4, n_examples=None):
    """
    Streamlit-friendly dataloader that accepts UploadedFile directly
    Handles Mac-created zips with __MACOSX folders
    
    Args:
        uploaded_file: Streamlit UploadedFile object (zip file)
        batch_size: int
        shuffle: bool
        num_workers: int
        n_examples: int, optional number of examples to limit
    
    Returns:
        DataLoader
    """
    import zipfile
    
    # Create BytesIO from uploaded file
    zip_bytes = io.BytesIO(uploaded_file.getvalue())
    
    # Create temp directory for extraction
    temp_dir = tempfile.mkdtemp()
    
    # Extract zip file, filtering out Mac metadata
    with zipfile.ZipFile(zip_bytes, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Skip Mac metadata files
            if '__MACOSX' in member or member.startswith('.'):
                continue
            zip_ref.extract(member, temp_dir)
    
    # Find the dataset folder, ignoring Mac artifacts
    extracted_items = [
        item for item in os.listdir(temp_dir) 
        if not item.startswith('.') and item != '__MACOSX'
    ]
    
    if len(extracted_items) == 0:
        raise ValueError("No valid files found in zip after filtering Mac metadata")
    
    # Determine dataset path
    # If single folder extracted, use it
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_dir, extracted_items[0])):
        dataset_path = os.path.join(temp_dir, extracted_items[0])
    else:
        # Multiple items or files - use temp_dir directly
        dataset_path = temp_dir
    
    # Use the existing get_dataloader function
    return get_dataloader(dataset_path, batch_size, shuffle, num_workers, n_examples)


def get_dataloader_smart(dataset_source, batch_size=32, shuffle=True, num_workers=4, n_examples=None):
    """
    Smart dataloader that accepts EITHER:
    - String path (for local testing)
    - Streamlit UploadedFile (for web app)
    
    Args:
        dataset_source: str path OR Streamlit UploadedFile object
        batch_size: int
        shuffle: bool
        num_workers: int
        n_examples: int, optional number of examples to limit
    
    Returns:
        DataLoader
    """
    # Check if it's a string path
    if isinstance(dataset_source, str):
        return get_dataloader(dataset_source, batch_size, shuffle, num_workers, n_examples)
    
    # Otherwise, assume it's a Streamlit UploadedFile
    else:
        return get_dataloader_from_upload(dataset_source, batch_size, shuffle, num_workers, n_examples)
