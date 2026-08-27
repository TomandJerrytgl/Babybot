# Jetson deployment

The service files use the confirmed Jetson account `Tom` and project location
`/home/Tom/Babybot`. Adjust `User`, `WorkingDirectory`, camera indices, and the
Python path if the installation moves.

Required Python imports are `numpy` and `cv2`. On Jetson, prefer the OpenCV
package supplied with JetPack when available.

## Stereo training recordings

The web page now controls paired stereo recording. Local datasets default to
`recordings/`; that directory is intentionally ignored by the public Babybot
repository. Each recording directly writes synchronized left and right videos
plus `pairs.csv` and `metadata.json`; it does not retain a duplicate JPEG frame
tree or frame archive.

GitHub visibility is repository-wide, so recordings cannot be private inside
the public Babybot repository. Prepare the private `TGLgeneral` clone for Tom:

```bash
git lfs install
git clone YOUR_PRIVATE_TGLGENERAL_URL /home/Tom/TGLgeneral
```

Configure SSH keys or another Git credential helper for that service user.
Never put a token in the service file or source code. Pass these arguments to
`record.py`:

```text
--recording-root /home/Tom/Babybot/recordings --upload-repo /home/Tom/TGLgeneral
```

The account needs write access to `TGLgeneral`, and the repository needs enough
Git LFS storage and bandwidth. Upload failures leave local data intact and can
be retried from the web page. A clone of the private dataset is directly
readable with `StereoDataset`, which synchronously decodes both videos for
training. Older v1 batches containing JPEG frames or `frames.zip` remain
readable.

## Runtime modes

Babybot has three independent entry points:

```bash
python3 awake.py       # live cognition, port 8080
python3 record.py      # capture and recording inspection, port 8081
python3 dreaming.py    # hardware-free offline processing, port 8082
```

`main.py` remains a compatibility entry point for Awake. Awake and Record both
own the stereo cameras, so do not run them together. Record never opens or
modifies Memory. Dreaming never opens camera hardware and can run on macOS,
Windows, Linux, or Jetson with Python, NumPy, and OpenCV.

Typical Jetson Record command:

```bash
python3 record.py --left-camera 0 --right-camera 2 --port 8081 \
  --recording-root /home/Tom/Babybot/recordings \
  --upload-repo /home/Tom/TGLgeneral
```

Typical local Dreaming command, including macOS:

```bash
python3 dreaming.py --port 8082 \
  --recordings /path/to/local/recordings \
  --memory /path/to/local/memory
```

Dreaming Memory is deliberately local in this stage. It is not automatically
pulled from or pushed to GitHub. Its SQLite index and sample files are protected
by transactions and a process lock. The feature-training boundary is present,
but no neural-network dependency or model is required yet.

After copying the project and checking that `python3 main.py` works manually:

```bash
sudo cp deploy/babybot.service /etc/systemd/system/babybot.service
sudo systemctl daemon-reload
sudo systemctl enable --now babybot.service
```

To install Record as a service instead, copy `babybot-record.service`. Awake and
Record must not be enabled at the same time because both require the cameras.

Inspect runtime logs with:

```bash
journalctl -u babybot.service -f
```

Stop the service gracefully with:

```bash
sudo systemctl stop babybot.service
```

The preview server listens only on Jetson's loopback interface. It is not
directly reachable through the Jetson Wi-Fi address.

With a VS Code Remote SSH connection, open the **Ports** panel, forward remote
port `8080`, and then open `http://127.0.0.1:8080` on the computer.

Alternatively, create the tunnel manually from the computer and keep the SSH
session open:

```powershell
ssh -L 8080:127.0.0.1:8080 Tom@JETSON_IP_ADDRESS
```

Then open `http://127.0.0.1:8080`. Closing the tunnel removes computer access
to the preview.
