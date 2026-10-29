# Acquisition: rig setup and debugging

Everything upstream of `moseq2-extract`. The analysis pipeline begins at a raw depth
recording; this covers producing one, and diagnosing a rig that will not.

## The three acquisition paths

Camera choice determines the software *and* the operating system. They are not
interchangeable.

| Camera | Acquisition software | OS | Depth computed on |
|---|---|---|---|
| Kinect for Xbox One (Kinect v2) | `dattalab/kinect2-nidaq` (C#/.NET 4.5.1) | **Windows** | **Host GPU** (DirectX 11) |
| Azure Kinect DK | `dattalab/azure-acquire` (Python) | **Ubuntu 18.04** | **Host GPU** (DirectX on Windows, OpenGL 4.4 on Linux) |
| Orbbec Femto Mega | `dattalab/orbbec-acquire` (Python) | **Ubuntu 22.04** | **On-camera** (built-in depth engine) |

`kinect2-nidaq` is a Windows GUI binary (`<OutputType>WinExe</OutputType>`). The two
Python tools declare `platforms=['mac','unix']` and the protocols paper marks their
Ubuntu versions CRITICAL. A user reporting Task Manager is therefore on the Kinect v2
path.

The Femto Mega's on-camera depth engine is the single most useful asymmetry: a host-GPU
depth failure is impossible on it, so any GPU hypothesis evaporates for that camera.

## Rig geometry

Published mounting heights, camera face to arena floor (Lin et al. 2024, *Nature
Protocols*; same numbers on the moseq2-app wiki):

- **Kinect v2: 673 ± 5 mm (26.5 in)**
- **Azure Kinect and Orbbec Femto Mega: 596 ± 5 mm (23.5 in)**
- Arena: US Plastic item 14317, 17 in diameter

Mount too low and the top of the animal crosses the sensor's minimum range, where IR
saturation destroys the time-of-flight phase measurement and those pixels are written as
**0**. The symptom is a missing patch on the animal's back during rears, worst on
light-coated animals. Kinect v2's floor is 500 mm with no user-accessible near mode;
Azure/Femto NFOV is 500 mm, WFOV 250 mm.

**Measure sensor face to floor, not box interior height.** Body depth along the optical
axis is ~67 mm for a Kinect v2, ~126 mm for an Azure Kinect DK, ~145 mm for a Femto Mega,
plus bracket and rear cable clearance. `max_height` defaults to 120 mm, so keeping the
whole animal outside the dead zone needs the sensor face at ≥ 620 mm — which a Kinect v2
cannot reach if its body hangs inside a box shorter than about 700 mm. Cut a ceiling port
so the body sits above the enclosure.

Mounting lower magnifies the animal (finer mm/px). Nothing in the analysis pipeline reads
absolute camera distance — `true_depth` and the floor plane are measured per session — but
apparent pixel size does affect `moseq2-pca`'s pixel-domain filters and the extraction
`crop_size`. See **Camera height and the analysis pipeline** below.

Reflectivity sets the *effective* minimum range, which is why the protocol treats surface
prep as CRITICAL: sand the bucket, especially the wall-floor junction, then ultra-flat
black paint. A glossy surface saturates further out than a matte one.

## Debugging: run the vendor's tool first

**The first move on any acquisition failure is a tool with none of the lab's code in it.**
This splits the problem in one action: if the vendor's own tool fails the same way,
nothing in `kinect2-nidaq` / `azure-acquire` / `orbbec-acquire` can be the cause, and no
application setting will help.

| Camera | Vendor tool | What it gives |
|---|---|---|
| Kinect v2 | **Kinect Configuration Verifier** (Start → SDK Browser 2.0) | Per-layer itemized rows **and** a live stream test |
| Kinect v2 | SDK Browser 2.0 → Color Basics-D2D / Depth Basics-D2D / Infrared Basics-D2D | Isolates color vs depth vs IR paths |
| Kinect v2 | Kinect Studio 2.0 | Independent capture path; distinguishes per-app access policy blocks |
| Azure Kinect | `k4aviewer` | A stream; with logging enabled, numbered depth-engine error codes |
| Femto Mega | OrbbecViewer, or `pyorbbecsdk/examples/depth_viewer.py` | A stream; the SDK logs verbosely to stdout by default |

Only the Kinect v2 has a tool that both partitions *and* labels the layer. For the others,
enable logging to get the equivalent detail:

```bash
# Azure Kinect
export K4A_ENABLE_LOG_TO_A_FILE=k4a.log
export K4A_LOG_LEVEL=i        # or 't' for trace
```

Depth-engine error codes (from `k4aplugin.h`) are more diagnostic than anything the
Verifier reports, just less discoverable:

| Code | Meaning |
|---|---|
| 108 | Depth engine plugin not loaded (missing `libdepthengine`/`depthengine_*.dll`) |
| 201 | GPU out of memory |
| 204 | GPU API returned failure — no usable graphics device |
| 206 | GPU shader compilation error — driver |
| 207 | OpenGL context creation error |
| 208 | GPU processing timed out |

### Kinect Configuration Verifier

Start → **SDK Browser 2.0 (Kinect for Windows)** → **Kinect Configuration Verifier**. It
ships with the **Kinect for Windows SDK 2.0**, which is a *separate download from the
Runtime 2.0*; both are listed in the install instructions, but a rig builder who installed
only the Runtime will not have this tool. Close any app holding the sensor first.

Rows, in order:

| Row | Reading |
|---|---|
| Update Configuration Definitions | Needs internet. On failure it falls back to a ~2014 whitelist — see below |
| Operating System, Processor Cores, Physical Memory | Green expected; any red means genuinely unsupported hardware |
| **Graphics Processor** | Green means Windows can hand the SDK a DirectX 11 device. **A pass rules out the entire GPU family of causes** |
| **USB Controller** | Green = recognized and supported. Orange = chipset unidentified, common and not itself alarming. Red = incompatible |
| Kinect Connected | Green if a device is enumerated |
| Verify Kinect Software Installed | Not important; Windows supplies drivers |
| **Verify Kinect Depth and Color Streams** | The live test. Green at a sustained **29-30 FPS** |

**The definitions row and the USB row are coupled.** The USB check compares the detected
controller against the downloaded definitions database. When the download fails, the tool
uses a whitelist from around 2014, so any modern controller reports as **"Unknown USB 3.0
port detected"** — meaning *newer than the database*, not *bad*. On a recent machine that
row is uninformative in both directions, and the controller must be identified by hand.

**The evidence is asymmetric.** A stream-test *failure* strongly rules out the lab's
software. A *pass* rules the lower stack back in only weakly, because the test is brief:
sustained-load faults (drops at minute 20, a cable degrading as it warms, contention that
appears once the disk writer runs) can pass it.

The Verifier also says nothing about the NI-DAQmx dependency, which is a separate and
common `kinect2-nidaq` failure — that one presents as the executable refusing to *launch*
("files required for the application are missing"), not as a capture problem.

## Second axis: initialization vs throughput

Orthogonal to "whose code", and the two together quarter the space.

**Check whether the depth file grows.**

| Observation | Diagnosis |
|---|---|
| `depth_<ts>.dat` stays at **0 bytes** | No frame ever arrived. Initialization / enumeration |
| Grows at **~13 MB/s**, but the UI is unresponsive | Frames flowing, writer cannot keep up. Throughput |
| Grows, session completes, gaps in `depth_ts.txt` | Dropped frames. CPU/disk contention, or an undersized GPU |

For dropped frames the lab's own test is the timestamp file: consecutive deltas in
`depth_ts.txt` should be ~30 (ms). More than 1% dropped means the acquisition computer
needs attention. Known contributors: Windows Superfetch, Windows Search, antivirus.

A disk cannot be too slow for zero bytes, so a first-frame freeze is never a throughput
problem.

### `kinect2-nidaq` status indicators are unreliable during a stall

During an initialization failure the interface reports false-healthy on every visible
element. `sensor.Open()` is asynchronous and nothing verifies a frame ever arrived, so the
frame-arrival callbacks simply never fire: no exception, no crash, window responsive, and
the display timer re-renders the last frame indefinitely.

- Stream lights are driven by the **checkboxes**, not by frame arrival
- "Frames Dropped" counts only null-frame events, so it reads `Color 0 Depth 0`
- The ETA countdown and "Recording" text advance on wall clock alone

**Only the on-disk file size is trustworthy.**

Unrelated but easily confused with a freeze: the shipped Min/Max depth defaults
(2000/5000) exceed the sliders' maximum of 1000 and coerce to a 1 mm window, so **the
depth preview renders solid black on a fresh install** regardless of whether frames are
arriving. Move the distance sliders to bracket the real camera height (about 450/705 mm
for a 700 mm mount) before judging the preview.

Useful isolation within `kinect2-nidaq`, in order: **Preview Mode** (opens no files, starts
no writer threads — camera and display only), then Preview Mode **with Color Stream
unchecked**, which removes the only GPU-backed call in the lab's code
(`CoordinateMapper.MapDepthFrameToColorSpace`) plus a 1.7 MB/frame allocation. MoSeq does
not use the color video, so running with color off is a valid production configuration.

## USB

The dominant failure surface for Kinect v2, because the sensor ships raw IR correlation
frames rather than finished depth and therefore **reserves ~2.47 Gbps of isochronous
bandwidth** — roughly **77% of a USB 3.0 controller's entire periodic budget** (80% of
4.0 Gbps payload = 3.2 Gbps) — while writing only 13 MB/s to disk. Bus demand is ~26×
disk demand. The reservation is negotiated **at stream start**, which is why contention
presents as a startup failure rather than degradation.

Microsoft documents both the failure and its folk remedy: *"Using a shared USB3 controller
may lead to resource contention which prevents the Kinect from initializing"*, and *"there
are other devices which are reserving bandwidth on your controller… try to reboot. For
integrated controllers, this will often solve the issue."* Separately, *"On first plugin,
the firmware on the device will be updated. This may result in device enumeration
happening several times in the first minute."* So "works after several restarts" and
"works if I wait" are documented behaviours, not mysteries.

**Host controller requirements differ by camera:**

- **Kinect v2**: *"Built-in USB 3.0 host controller (Intel or Renesas chipset)"* — Intel or
  Renesas only. Add-in cards must support Gen-2.
- **Azure Kinect on Windows**: *"Intel, Texas Instruments (TI), and Renesas are the only
  host controllers that are supported."*

Do not carry TI across to the Kinect v2; it appears only in the Azure documentation.
**Renesas and NEC are the same silicon** (Renesas absorbed NEC Electronics in 2010; the
µPD720201/720202 ship under both names), so an NEC entry is a pass. **ASMedia is known not
to work** — attested by libfreenect2 (*"ASMedia controllers are known to not work"*) and by
Microsoft's Azure documentation (ASM1142), though not by the Kinect v2 page, which names no
bad vendors. VIA and Fresco Logic are community-reported as incompatible.

**Identify the controller by hand**, since the Verifier often cannot:

```
Device Manager → expand "Universal Serial Bus controllers"
  → read the "eXtensible Host Controller" entries for vendor
```

Expanding that one node and reading the vendor names is usually conclusive and is far
easier than the Devices-by-connection tree. Only when several vendors appear is the tree
needed, to establish which controller the camera actually sits on:

```
Device Manager → View → Devices by connection → locate the camera
```

**Practical port guidance:**

- Prefer a **10 Gbps port** (USB 3.1 Gen 2 / USB 3.2 Gen 2). The isochronous reservation is
  ~77% of a 5 Gbps controller's periodic budget but only ~38% of a 10 Gbps one. Port speed
  class and controller vendor are independent questions — a blue 5 Gbps port is genuinely
  USB 3.0, and can still be the wrong place to plug in.
- Rear motherboard ports over front-panel, which are frequently behind an internal hub.
- Unplug other devices from the same controller. Webcams, audio interfaces and second
  cameras all reserve isochronous bandwidth.
- Kinect v2 requires its powered adapter; power-cycling that brick clears a stuck device
  without rebooting the host, which also distinguishes a stuck camera from a stuck OS.
- Extension and repeater cables are a documented risk. The BOM lists a 25 ft Tripp Lite
  U328-025 active repeater, while the same wiki page warns *"we've had mixed results using
  USB 3.0 extension cords with the Kinect 2, to be safe you will want to use the stock USB
  3.0 cable."* Test with the stock cable first.
- Azure/Femto over USB-C: passive cables ≤ 1.5 m, active beyond that.

### Composite-device enumeration

The Kinect v2 presents camera and microphone array as one composite USB device. A healthy
install shows all of these in Device Manager: `Microphone Array (Xbox NUI Sensor)`,
`KinectSensor Device`, `Xbox NUI Sensor`, and `Kinect for Windows USB Audio`. **Missing
audio nodes indicate partial enumeration and point at USB**, not at audio. Comparing the
node list when frozen against when working is a good confirmatory test.

Related and cheap: on Windows 10/11, audio-enhancement DSP on the Kinect microphone array
can reset the composite device mid-stream. Sound → More sound settings → Recording →
Properties → Advanced → untick **Enable audio enhancements**, for *every* Kinect entry.
Immediate, no reboot.

Also verify only a **single instance of `KinectService`** is running. When frozen,
`net stop KinectMonitor` followed by `net start KinectMonitor` from an elevated prompt
recovers a stuck service without a reboot, and distinguishes service state from hardware.

## Data rates and storage

Arithmetic, for sizing a rig and for ruling throughput in or out.

| Configuration | MB/s | 30-min session |
|---|---|---|
| Kinect v2 depth (what MoSeq records) — 512×424×2 B×30 fps | **13.03** | **23.5 GB** |
| Kinect v2 color 1920×1080 YUY2 (recorded as mp4, optional) | 124.4 raw | — |
| Azure / Femto depth+IR, 640×576×2 B×30 fps ×2, pre-compression | **44.24** | 79.6 GB |

Depth-only acquisition uses roughly 10% of a 7200 rpm HDD and 3% of USB 3.0's practical
throughput, so storage is essentially never the cause of an acquisition failure. Sizing
still matters: ~281 GB/day at 12 sessions, and the 10 hours the paper gives as a modeling
minimum is ~469 GB.

Record color only when actually needed. It is not used by the analysis pipeline and the
protocol warns it *"can lead to a high number of dropped frames."*

## Camera height and the analysis pipeline

Changing mounting height does not require reconfiguring the analysis pipeline, because the
geometry is measured rather than assumed:

- **`true_depth` is measured per session** as the median background depth inside the
  detected ROI. The `673.1` appearing in several function signatures is never reached
  through the normal extraction path.
- **`bg_roi_depth_range` auto-adjusts per session** to ±50 mm around the floor depth at the
  arena centre. The configured value is used only when `manual_set_depth_range` is set.
- `min_height` / `max_height` are relative to the fitted floor plane, and the background is
  a per-pixel median, so both adapt to any distance.
- `moseq2-model` contains no reference to camera geometry at all.

What *does* change is apparent pixel size. A lower camera magnifies the animal, so check:

- **`crop_size`** (default 80×80 px, ~130 mm at the Kinect v2 reference height). An
  overflowing crop is silently dropped as an all-zero frame, with no warning.
- **`moseq2-pca`'s pixel-domain filters**, `gaussfilter_space` and `tailfilter_size`. Scale
  both by the magnification ratio. The existing `--camera-type azure` preset does exactly
  this at ~1.5×, and is a useful sanity check on magnitude.

Do not pool sessions recorded at different heights into one PCA or one AR-HMM. PCs live in
a fixed pixel basis with no size normalization, and whitening does not absorb a
magnification change, so the model will segment on cohort identity. Size-normalize the
extracted frames to a common apparent scale first.

## Errata in published documentation

Points where the lab's own materials are wrong or incomplete, worth knowing before quoting
them at anyone.

- **The kinect2-nidaq wiki says to avoid USB "3.1" ports.** Under pre-2017 naming a 10 Gbps
  port *is* USB 3.1 Gen 2, and those are the better choice for this camera. The advice
  points away from the ports with the most isochronous headroom.
- **No lab documentation states the host-controller requirement.** The published spec asks
  only for "at least 1 USB 3.0 port", so a rig can satisfy every stated requirement and
  still sit on an unsupported controller or behind a front-panel hub.
- **"A single 20-minute, uncompressed depth video is ~20 GB"** overstates Kinect v2 by
  ~28%; the true figure is 15.6 GB, and ~20 GB is closer to the Azure/Femto depth-only
  case.
- **`generate-config` differs between CLI and notebook.** The CLI azure preset sets
  `crop_size` and `camera_type`; the notebook equivalent omits both, so a notebook user
  selecting azure silently gets an 80×80 crop and `camera_type: auto`.
- **Microsoft never supported this SDK past Windows 8.1.** The Kinect v2 system
  requirements list Windows 8 / 8.1 (x64) only. Windows 10 and 11 rigs work but are outside
  the stated support envelope, which is where the audio-enhancement regression lives.

## Sources

- Lin et al. 2024, *Characterizing the structure of mouse behavior using Motion Sequencing*,
  Nature Protocols — https://pmc.ncbi.nlm.nih.gov/articles/PMC11552546/
- Kinect v2 system requirements —
  https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn782036(v=ieb.10)
- Kinect v2 getting started / troubleshooting —
  https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn782035(v=ieb.10)
- Kinect v2 SDK known issues —
  https://learn.microsoft.com/en-us/previous-versions/windows/kinect/dn758677(v=ieb.10)
- Azure Kinect troubleshooting —
  https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/troubleshooting
- Azure Kinect hardware specification —
  https://learn.microsoft.com/en-us/previous-versions/azure/kinect-dk/hardware-specification
- Configuration Verifier per-row reference (third-party) —
  https://docs.velogicfit.com/section-01-getting-started/kinect-troubleshooting
- Windows 11 audio-enhancement fix —
  https://brekel.com/kinect-disconnect-fix/ and
  https://learn.microsoft.com/en-us/answers/questions/5609514/kinect-for-windows-11-has-broken-solve-found
- libfreenect2 USB notes — https://github.com/OpenKinect/libfreenect2
- MoSeq cage and BOM — https://github.com/dattalab/moseq2-app/wiki/MoSeq-Cage
