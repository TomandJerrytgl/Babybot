# Jetson deployment

The service file assumes that this project is installed at `/opt/babybot` and
runs as a Linux user named `babybot`. Adjust `User`, `WorkingDirectory`, camera
indices, and the Python path if the Jetson uses different values.

Required Python imports are `numpy` and `cv2`. On Jetson, prefer the OpenCV
package supplied with JetPack when available.

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
