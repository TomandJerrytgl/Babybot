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

From a computer on the same Wi-Fi network, open
`http://JETSON_IP_ADDRESS:8080`.
