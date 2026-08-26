# Jetson deployment

The service file assumes that this project is installed at `/opt/babybot` and
runs as a Linux user named `babybot`. Adjust `User`, `WorkingDirectory`, camera
indices, and the Python path if the Jetson uses different values.

Required Python imports are `numpy` and `cv2`. On Jetson, prefer the OpenCV
package supplied with JetPack when available.

## Stereo training recordings

The web page now controls paired stereo recording. Local datasets default to
`recordings/`; that directory is intentionally ignored by the public Babybot
repository. Each recording directly writes synchronized left and right videos
plus `pairs.csv` and `metadata.json`; it does not retain a duplicate JPEG frame
tree or frame archive.

GitHub visibility is repository-wide, so recordings cannot be private inside
the public Babybot repository. Prepare a private `TGLgeneral` clone owned by the
`babybot` service user instead:

```bash
sudo -u babybot git lfs install
sudo -u babybot git clone YOUR_PRIVATE_TGLGENERAL_URL /opt/TGLgeneral
```

Configure SSH keys or another Git credential helper for that service user.
Never put a token in the service file or source code. Then add these arguments
to `ExecStart`:

```text
--recording-root /opt/babybot-data/recordings --upload-repo /opt/TGLgeneral
```

The account needs write access to `TGLgeneral`, and the repository needs enough
Git LFS storage and bandwidth. Upload failures leave local data intact and can
be retried from the web page. A clone of the private dataset is directly
readable with `StereoDataset`, which synchronously decodes both videos for
training. Older v1 batches containing JPEG frames or `frames.zip` remain
readable.

After copying the project and checking that `python3 main.py` works manually:

```bash
sudo cp deploy/babybot.service /etc/systemd/system/babybot.service
sudo systemctl daemon-reload
sudo systemctl enable --now babybot.service
```

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
