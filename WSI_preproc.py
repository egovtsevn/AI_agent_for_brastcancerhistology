import os
import shutil
import threading
import numpy as np
import queue
import torch
import timm
from openslide import OpenSlide
from remotezip import RemoteZip
from concurrent.futures import ThreadPoolExecutor, as_completed
import torchvision.transforms.v2.functional as F
from shapely.geometry import Polygon, box
from shapely.strtree import STRtree
from wholeslidedata.annotation.wholeslideannotation import WholeSlideAnnotation

images_zenodo_url = 'https://zenodo.org/records/16812932/files/images.zip?download=1' # BEETLE train dataset URL
temp_dir = './temp_slides' # temp dir for downloaded WSIs
model_drive_path = "..." # a path to model weights
embeddings_drive_path = "..." # a path where embeddings will be put
annotations_path = "..." # a path to annotations
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(embeddings_drive_path, exist_ok=True)

# Model initialization
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)
model.load_state_dict(torch.load(os.path.join(model_drive_path, "pytorch_model.bin"), map_location="cpu"), strict=True)
model.to("cuda")
model.eval()

# Retrieval of WSI file names
with RemoteZip(images_zenodo_url) as zip_file:
    tif_files = [file for file in zip_file.namelist() if file.endswith(".tif")]

def images_download(tif_files, images_zenodo_url, temp_dir, download_queue):
    """Sequential download of WSI files from BEETLE repository
    tif_files: list of WSI file names
    images_zenodo_url: BEETLE train dataset URL
    temp_dir: temp dir for WSIs
    download_queue: queu of WSIs for sequential processing"""
    try:
        with RemoteZip(images_zenodo_url) as zip_file:
            for tif_file in tif_files:
                xml_path = os.path.join(
                    annotations_path,
                    f"{os.path.splitext(os.path.basename(tif_file))[0]}.xml")
                if not os.path.exists(xml_path):
                  print(f"file {xml_path} wasnt found")
                  continue
                base_name = os.path.splitext(os.path.basename(tif_file))[0]
                expected_save_name = f"{base_name}_embeddings.npz"
                if os.path.exists(os.path.join(embeddings_drive_path, expected_save_name)):
                    print(f"WSI {base_name} was already processed")
                    continue
                try:
                    image_drive_path = os.path.join(temp_dir, os.path.basename(tif_file))
                    with zip_file.open(tif_file) as source, open(image_drive_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    download_queue.put(image_drive_path)
                    print(f"{os.path.basename(image_drive_path)} downloaded")
                except Exception as e:
                    print(f"Downloading error: {tif_file}: {e}")
    except Exception as e:
        print(f"Error connectiong to the archive: {e}")
    finally:
        download_queue.put(None)

def patch_has_annotation(x, y, annotation_tree, annotation_polygons, patch_size):
    """
    Returnes True if patch (x, y) crooses XML-annotation
    x, y: coordinates of patch upper left pixel
    annotation_tree: STRtree for XML-polygons.
    annotation_polygons: polygons souurce list.
    patch_size: patch size in pixels.
    """

    if annotation_tree is None:
        return False

    patch_box = box(x, y, x + patch_size, y + patch_size,)
    candidates = annotation_tree.query(patch_box)
    for candidate in candidates:
        if isinstance(candidate, (int, np.integer)):
            poly = annotation_polygons[int(candidate)]
        else:
            poly = candidate
        if poly.intersects(patch_box):
            return True

    return False

thread_local = threading.local()

def get_thread_slide(image_path): # creates a separate slide handler for each thread
    if not hasattr(thread_local, "slide") or thread_local.slide_path != image_path:
        if hasattr(thread_local, "slide") and thread_local.slide is not None:
            thread_local.slide.close()
        thread_local.slide = OpenSlide(image_path)
        thread_local.slide_path = image_path
    return thread_local.slide

def read_single_patch(image_path: str, x: int, y: int, patch_size: int = 224):
    slide = get_thread_slide(image_path)
    patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
    tensor_patch = F.to_image(patch)
    return tensor_patch, x, y

def image_cut(image_path, patch_size=224, mask_level=None, max_workers=4, grid = 20):
    """
    Cuts WSI into patches for further processing
    grid: additional area surrounding patch
    """

    slide = OpenSlide(image_path)
    slide_width, slide_height = slide.dimensions
    slide.close()

    xml_path = os.path.join(
        annotations_path,
        f"{os.path.splitext(os.path.basename(image_path))[0]}.xml"
    )

    wsa = WholeSlideAnnotation(xml_path)

    annotation_polygons = []
    for ann in wsa.annotations:
        try:
         poly = Polygon(ann.coordinates)

         if poly.is_valid and not poly.is_empty:
            annotation_polygons.append(poly)

        except Exception:
            continue

    annotation_tree = (
        STRtree(annotation_polygons)
        if annotation_polygons
        else None)

    all_coords = set()
    for i in range(0, slide_width - patch_size + 1, patch_size):
        for j in range(0, slide_height - patch_size + 1, patch_size):
            if patch_has_annotation(i, j, annotation_tree, annotation_polygons, patch_size):
                min_x = i - (grid // 2) * patch_size
                min_y = j - (grid // 2) * patch_size
                for n in range (grid):
                  for k in range (grid):
                    px = min_x + n * patch_size
                    py = min_y + k * patch_size
                    if 0 <= px <= slide_width - patch_size and 0 <= py <= slide_height - patch_size:
                        all_coords.add((px, py))

    # Processing of patches by chunks
    all_coords = list(all_coords)
    chunk_size = 256
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for k in range(0, len(all_coords), chunk_size):
            chunk = all_coords[k : k + chunk_size]
            futures = [
                executor.submit(read_single_patch, image_path, x, y, patch_size)
                for x, y in chunk
            ]
            for fut in as_completed(futures):
                patch, x, y = fut.result()
                yield patch, (x, y)

def disc_download(disc_download_queue):
  """ Saving of processing results """
  while True:
    smth = disc_download_queue.get()
    if smth is None:
        disc_download_queue.task_done()
        break
    image_drive_path, current_slide_results = smth
    coords_arr = np.array([res[0] for res in current_slide_results])
    embs_arr = np.array([res[1] for res in current_slide_results])
    save_name = f"{os.path.splitext(os.path.basename(image_drive_path))[0]}_embeddings.npz"
    save_path = os.path.join(embeddings_drive_path, save_name)
    local_save_path = os.path.join("/content", save_name)
    np.savez_compressed(local_save_path, coords=coords_arr, embeddings=embs_arr)
    shutil.move(local_save_path, save_path)
    print(f"Successfully saved: {save_name} (patches: {len(coords_arr)})")
    disc_download_queue.task_done()


def gpu_worker(patch_queue, disc_download_queue, model, batch_size=32): # GPU thread
    with torch.inference_mode():
        batch_tensors = []
        batch_coords = []
        results = []

        def process_batch():
            if not batch_tensors:
                return
            batch_cpu = torch.stack(batch_tensors).pin_memory()
            batch_gpu = batch_cpu.to("cuda", non_blocking=True)
            inp = batch_gpu.float().div(255.0)
            inp = F.normalize(inp, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            embs = model(inp).cpu().numpy()
            for c, e in zip(batch_coords, embs):
              results.append((c, e))
            batch_tensors.clear()
            batch_coords.clear()

        while True:
            item = patch_queue.get()

            if item is None:
                process_batch()
                patch_queue.task_done()
                break

            if item == 'FLUSH':
                process_batch()
                image_drive_path = image_name_queue.get()
                disc_download_queue.put((image_drive_path, results))
                results = []
                image_name_queue.task_done()
                patch_queue.task_done()
                continue

            tensor_patch, coords = item
            batch_tensors.append(tensor_patch)
            batch_coords.append(coords)

            if len(batch_tensors) >= batch_size:
                process_batch()

            patch_queue.task_done()

# Initialization of ques an GPU-thraed
preproc_queue = queue.Queue(maxsize=1)
patch_queue = queue.Queue(maxsize=128)
disc_download_queue = queue.Queue()
image_name_queue = queue.Queue()

gpu_thread = threading.Thread(
    target=gpu_worker,
    args=(patch_queue, disc_download_queue, model, 64),
    name="GPU-Inference-Thread"
)
gpu_thread.start()

disc_download_thread = threading.Thread(
    target=disc_download,
    args=(disc_download_queue,),
    name="Disc-Download-Thread"
)
disc_download_thread.start()

# Main cykle of WSI processin
with ThreadPoolExecutor(max_workers=1) as download_executor:
    download_executor.submit(images_download, tif_files, images_zenodo_url, temp_dir, preproc_queue)

    while True:
        image_drive_path = preproc_queue.get()
        if image_drive_path is None:
            preproc_queue.task_done()
            break
        image_name_queue.put(image_drive_path)
        try:
            gen = image_cut(image_drive_path, patch_size=224, max_workers=4)
            for patch, coords in gen:
                patch_queue.put((patch, coords))

            patch_queue.put('FLUSH')
            patch_queue.join()

        except Exception as e:
            print(f"Error of processing {image_drive_path}: {e}")
        finally:
            if os.path.exists(image_drive_path):
                os.remove(image_drive_path)
            preproc_queue.task_done()

patch_queue.put(None)
gpu_thread.join()
disc_download_queue.put(None)
disc_download_thread.join()
print("Processing is complrtr!")