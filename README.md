# Unsupervised ReID

## Task A — Person Identity Catalogue
> Identify and unify all individuals appearing in the 4 clips under a consistent global ID.

#### Pipeline overview:
- Key-frame extraction (Adaptive / Top-K / All)
- Person detection & tracking (YOLO v11n-seg)
- Cropping + segmentation mask generation
- Embedding extraction (Torchreid – OSNet x1.0)
- Clustering (OPTICS / DBSCAN)
- Logical unification of Local IDs → Global IDs

## Notes & Recommendations

- Ensure consistent clip naming (1.mp4–4.mp4).
- Use GPU runtime if available (Colab → Runtime → Change runtime type → GPU).
- All randomness is fixed with np.random.seed(42) for reproducibility.
