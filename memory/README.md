# Babybot visual memory

This directory is populated by `main.py` while Babybot learns. Runtime files
(`manifest.json`, object directories, JPEG samples and metadata) are intentionally
ignored by Git because they are observations learned by one robot in one environment.

The first version stores at most 20 distinct visual objects and at most three diverse
stereo samples per object. Reaching 20 objects stops learning only; camera, perception,
attention and web preview continue running.
