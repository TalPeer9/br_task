import os
from collections import defaultdict
from glob import glob
import json
import random
import time
from datetime import datetime

import yaml
from tqdm import tqdm

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN, OPTICS

import av
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.core.display_functions import display

import torch
import cv2
from ultralytics import YOLO
import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchreid.reid.utils import FeatureExtractor

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

num_clips = 4
topk = True
k = - 0.01
save_to_output = True

track_conf = 0.75
track_classes = [0]
track_stride = 1

root_dir = "/kaggle/working/keyframes"
frames_out_dir = "/kaggle/working/keyframes"
clips_ids = range(1, num_clips + 1)


def extract_key_frames_adaptive(video_path, max_frames=500, k=1.25, save_to_output=False):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading video {video_path}")

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diffs = []
    frames_gray = []
    orig_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        orig_frames.append(frame)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
        diff_sum = np.sum(frame_diff)
        diffs.append(diff_sum)
        frames_gray.append((frame, diff_sum))
        prev_frame_gray = frame_gray

    cap.release()

    if len(diffs) == 0:
        return []

    mean_val = np.mean(diffs)
    std_val = np.std(diffs)
    threshold = mean_val + (k * std_val)

    orig_frames_count = len(orig_frames)
    key_frames_grey = [i for i, (frame, d) in enumerate(frames_gray) if (d >= threshold or i == 0)]
    key_frames = []

    for i in key_frames_grey:
        key_frames.append(orig_frames[i])
    print(
        f"Original frames count: {orig_frames_count} | output frames count: {len(key_frames)} (dropped {orig_frames_count - len(key_frames)} frames.)")
    if len(key_frames) > max_frames:
        key_frames = key_frames[:max_frames]

    if save_to_output:
        saved_paths = save_key_frames(video_path, key_frames, out_dir=frames_out_dir)
    return key_frames


def extract_key_frames_topk(video_path, num_key_frames=500, save_to_output=False, k=None):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError(f"Error reading video {video_path}")

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diffs = []
    frames_all = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
        diff_val = np.mean(frame_diff) 
        diffs.append((diff_val, frame_idx, frame))
        prev_frame_gray = frame_gray
        frame_idx += 1

    cap.release()

    if len(diffs) == 0:
        return []

    diffs_sorted = sorted(diffs, key=lambda x: x[0], reverse=True)
    topk = diffs_sorted[:num_key_frames]
    topk_sorted_by_frame = sorted(topk, key=lambda x: x[1])
    key_frames = [frame for _, _, frame in topk_sorted_by_frame]

    print(
        f"Original frames count: {frame_idx + 1} | output frames count: {len(key_frames)} (dropped {frame_idx + 1 - len(key_frames)} frames.)")

    if save_to_output:
        saved_paths = save_key_frames(video_path, key_frames, out_dir=frames_out_dir)
    return key_frames


def save_key_frames(video_path, key_frames, out_dir):
    video_id = os.path.basename(video_path).split(".")[0]
    out_dir = f"{out_dir}/{video_id}"

    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for idx, frame in enumerate(key_frames):
        frame_path = os.path.join(out_dir, f"{video_id}_keyframe_{idx:03d}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_paths.append(frame_path)
    return saved_paths


def show_frame_ipython(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    image_bytes = encoded_image.tobytes()
    display(Image(data=image_bytes))


def get_clip(clip_id):
    clip_path = f"/kaggle/input/br-videos/{clip_id}.mp4"
    return clip_path


def show_image(img, resize_scale=0.5):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (int(w * resize_scale), int(h * resize_scale)))
    success, encoded_image = cv2.imencode('.jpg', img_resized)
    display(Image(data=encoded_image.tobytes()))


def create_key_frames(num_clips, k=1.5, save_to_output=False, topk=False):
    db_key_frames = {f"{clip_id}": None for clip_id in clips_ids}
    if topk:
        extractor = extract_key_frames_topk
    else:
        extractor = extract_key_frames_adaptive
    for clip_id in clips_ids:
        video_path = get_clip(clip_id=clip_id)
        db_key_frames[f"{clip_id}"] = extractor(video_path, k=k, save_to_output=save_to_output)
    return db_key_frames

def key_frames_tracker(model_ver = "yolo11n.pt", conf_file = None, num_clips = 4,show_detection=False):
    model = YOLO(model_ver)
    
    clips_ids = range(1, num_clips+1)
    db_tracks = {f"{clip_id}":None for clip_id in clips_ids}
    db_crops = {f"{clip_id}":None for clip_id in clips_ids}
    id_crops = defaultdict(lambda: [])
    id_masks = defaultdict(lambda: [])
    seen_tracks = set()
    
    for clip_id in clips_ids:
        frames_dir = os.path.join(root_dir, str(clip_id))
        frame_files = sorted(glob(os.path.join(frames_dir, "*.jpg")))
        print(f"Clip {clip_id}: Found {len(frame_files)} frames.")
        
        track_history = defaultdict(lambda: [])
        
        
        for frame_idx, frame_path in enumerate(frame_files):
            frame = cv2.imread(frame_path)
            result = model.track(frame, persist = True,
                                 conf = track_conf, classes = track_classes,
                                 vid_stride = track_stride, verbose = False,iou=yolo_iou,
                                    tracker=conf_file)[0]
            if result.boxes and result.boxes.is_track:
                boxes = result.boxes.xywh.cpu()
                boxes_xyxy = result.boxes.xyxy.cpu().numpy().astype(int) 
                track_ids = result.boxes.id.int().cpu().tolist()
                track_cls = result.boxes.cls.int().cpu().tolist()
                track_confs = result.boxes.conf.float().cpu().tolist()
                masks = result.masks.data.cpu().numpy() 

                new_tracking_flag = not seen_tracks.issuperset(set(track_ids))
                
                for box,crop_coords, mask, track_id, cls in zip(boxes,boxes_xyxy, masks,track_ids,track_cls):
                    if cls in [0]: # currently just persons
                        x1, y1, x2, y2 = crop_coords
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y))) 
                        
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            crops = id_crops[track_id]
                            crops.append(crop) 

                        if mask is not None:
                            mask_full = (mask > 0.5).astype(np.uint8)
                            mask_resized = cv2.resize(mask_full, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            black_bg = np.zeros_like(frame, dtype=np.uint8)
                            person_black_bg = np.where(mask_resized[..., None] == 1, frame, black_bg)
                            crop_masked = person_black_bg[y1:y2, x1:x2]
                            if crop_masked.size > 0:
                                id_masks[track_id].append(crop_masked)
                                
                if new_tracking_flag and show_detection:
                    frame = result.plot()
                    print(f"seen_tracks: {seen_tracks}")
                    print(f"new_tracks: {set(track_ids).difference(seen_tracks)}")
                    seen_tracks = seen_tracks.union(track_ids)
                    print(f"seen_tracks updated: {seen_tracks}")
                    show_image(frame)
        

        db_tracks[f"{clip_id}"] = track_history
    return db_tracks, id_crops,id_masks


def assign_model_conf(model_conf):
    tracker_type = model_conf["tracker_type"]
    track_high_thresh = model_conf["track_high_thresh"]
    track_low_thresh = model_conf["track_low_thresh"]
    new_track_thresh = model_conf["new_track_thresh"]
    track_buffer = model_conf["track_buffer"]
    match_thresh = model_conf["match_thresh"]
    fuse_score = model_conf["fuse_score"]
    proximity_thresh = model_conf["proximity_thresh"]
    appearance_thresh = model_conf["appearance_thresh"]
    with_reid = model_conf["with_reid"]
    model = model_conf["model"]


    yaml_content = f"""
    tracker_type: {tracker_type}
    track_high_thresh: {track_high_thresh}
    track_low_thresh: {track_low_thresh}
    new_track_thresh: {new_track_thresh}
    track_buffer: {track_buffer}
    match_thresh: {match_thresh}
    fuse_score: {fuse_score}
    gmc_method: None
    proximity_thresh: {proximity_thresh}
    appearance_thresh: {appearance_thresh}
    with_reid: {with_reid}
    model: {model}
    """

    with open("botsort_custom.yaml", "w") as f:
        f.write(yaml_content)


def assign_model_conf(filename="botsort_custom.yaml", **kwargs):
    """

    :param filename:
    :param kwargs:
    :return:
    """
    with open(filename, "w") as f:
        yaml.dump(kwargs, f, sort_keys=False)


extractor = FeatureExtractor(
    model_name="osnet_x1_0",
    model_path=None,
    device="cpu"
)


def get_embedding(crop):
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    feat = extractor([img_rgb])[0].cpu().numpy()
    feat = feat / np.linalg.norm(feat)
    return feat


def unify_crops(id_crops):
    return pd.DataFrame.from_records(id_crops).groupby(0)[1].apply(list).reset_index()


def get_crops_embeddings(id_crops):
    rows = []
    for person_id, person_crops in id_crops.items():
        embs = [get_embedding(c) for c in person_crops]

        mean_emb = np.mean(embs, axis=0)
        max_emb = np.max(embs, axis=0)

        rows.append({
            "person_id": person_id,
            "embedding": mean_emb,
            "max_embedding": max_emb,
        })

    return pd.DataFrame(rows)


def get_crops_embeddings_single(id_crops):
    rows = []
    for person_id, person_crops in id_crops.items():
        for c in person_crops:
            c_embedding = get_embedding(c)

            rows.append({
                "person_id": person_id,
                "embedding": c_embedding
            })

    return pd.DataFrame(rows)


def cluster_embeddings(df, alg='DBSCAN', cluster_metric='cosine', eps=0.25, min_samples=2, min_size_method="median"):
    X = np.vstack(df["embedding"].values)
    if alg == "DBSCAN":
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=cluster_metric).fit(X)
    elif alg == "OPTICS":
        min_c_size = calc_cluster_size(df, method=min_size_method)
        clustering = OPTICS(min_samples=min_samples, min_cluster_size=min_c_size, metric=cluster_metric,
                            max_eps=eps).fit(X)
    else:
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=cluster_metric).fit(X)
    df = df.copy()
    df["person_id_cluster"] = clustering.labels_
    return df.drop(columns=["embedding"])


def calc_cluster_size(df, calc_by="person_id", method="median", q=0.25):
    if method == "median":
        return int(df[calc_by].value_counts().median())
    elif method == "min":
        return int(df[calc_by].value_counts().min())
    elif method == "mean":
        return int(df[calc_by].value_counts().mean())
    elif method == "quantile":
        return int(df[calc_by].value_counts().quantile(q))
    else:
        return int(df[calc_by].value_counts().min())


def plot_crops_clustering(person_crops, cluster, person_id):
    if len(person_crops) > 10:
        sampled_crops = random.sample(person_crops, 10)
    else:
        sampled_crops = person_crops

    fig, axes = plt.subplots(1, len(sampled_crops), figsize=(3 * len(sampled_crops), 3))
    if len(sampled_crops) == 1:
        axes = [axes]
    for ax, crop in zip(axes, sampled_crops):
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    plt.suptitle(f"Cluster {cluster} | Person ID {person_id}", fontweight="bold", fontsize=14)
    plt.show()


def display_clusters(id_crops, chosen_clusters):
    for cluster_id, persons_ids_list in chosen_clusters.items():
        if cluster_id == -1:
            continue
        for person_id in persons_ids_list:
            person_crops = id_crops[person_id]
            plot_crops_clustering(person_crops, cluster_id, person_id)


def experiment_clustering(crops_embeddings_df):
    epses = np.linspace(0.15, 0.2, num=10)
    for eps in epses:
        tracks_df = cluster_embeddings(crops_embeddings_df, eps=eps, cluster_metric="cosine")
        print(f"================== eps {eps} ==================")
        print(tracks_df.groupby("person_id_cluster").person_id.value_counts())


def main():

    yaml_content = """
    tracker_type: botsort
    with_reid: True
    model: auto
    track_high_thresh: 0.65
    track_low_thresh: 0.5
    new_track_thresh: 0.1
    track_buffer: 200
    match_thresh: 0.85
    fuse_score: False
    gmc_method: None
    proximity_thresh: 0.5
    appearance_thresh: 0.8

    """


    with open("botsort_custom.yaml", "w") as f:
        f.write(yaml_content)

    print("botsort_custom.yaml saved!")

    versions = ["yolo11n.pt","yolo11n-pose.pt","yolo11n-seg.pt","yolo11n-obb.pt"]
    yolo_ver = versions[2]
    conf_file = "botsort_custom.yaml"

    db_tracks, id_crops,id_masks = key_frames_tracker(model_ver = yolo_ver,conf_file =conf_file,show_detection=True)
    print("====================")
    persons_ids = id_crops.keys()
    print(f"Detected {len(persons_ids)} persons: \n\t {persons_ids}")
    print("====================\n")

    print("Starting Embeddings creation..")
    crops_embeddings_df = get_crops_embeddings_single(id_crops,add_extra_features=True)
    print("\t Done.")

    run_experiments = False

    if run_experiments:
        experiment_clustering(crops_embeddings_df)

    else:
        _eps = 0.15 # 0175
        _min_samples = 30 # 50
        _min_size_method = "quantile" # median
        _apply_pca = False # false
        _n_components = 256

        tracks_df = cluster_embeddings(crops_embeddings_df,alg = " OPTICS", 
                                    eps=_eps,cluster_metric="cosine",
                                    min_samples=_min_samples,min_size_method=_min_size_method,
                                    apply_pca=_apply_pca,n_components=_n_components)
        cluster_gb = tracks_df.groupby('person_id_cluster').person_id.apply(set).to_dict()

        counts = tracks_df.groupby(["person_id"]).person_id_cluster.value_counts().rename("count").reset_index()
        counts = counts[~counts["person_id_cluster"].isin([0,-1])]
        max_cluster_gb = counts.loc[counts.groupby("person_id")["count"].idxmax(),["person_id","person_id_cluster"]].groupby("person_id_cluster").person_id.apply(list).to_dict()

        chosen_clusters = cluster_gb 
        display_clusters(chosen_clusters)
