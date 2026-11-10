# ExamRoom

ExamRoom is a Python application for camera-assisted classroom exam monitoring. It can analyze the whole classroom, sending alert to the teacher when it inspects a suspicious move. It analyzes camera frames for configurable behavior cues, records short evidence clips, and presents alerts
to a teacher for review. **An alert is not a finding of misconduct:** a teacher decides whether
to confirm or dismiss it.

The project includes a simulated room for trying the dashboard without a camera, a standalone
camera-and-dashboard mode, and optional features such as a roster, a seat map, a second camera,
and an edge/server deployment.

## What is included

- Face tracking and behavior rules for head turns, repeated glances, looking down, eye gaze,
  possible talking, and possible hand passing.
- Optional YOLO-based visible-phone detection.
- A Flask dashboard with sign-in, live camera view, pending alerts, evidence clips, incident
  history, teacher decisions, and an audit log.
- Encrypted evidence storage and configurable retention.
- Optional student roster and face-match suggestions, plus teacher-confirmed spreadsheet marks.
- A seat map that associates detected faces with approximate seats.
- System checks, performance profiles, tuning and evaluation tools, and a guided self-test.
- Optional second-camera and MQTT edge/server modes.

These are cues for human review, not reliable proof of cheating. See
[Limitations and responsible use](#limitations-and-responsible-use) before using cameras or
student data.

## Requirements

- Python 3.10 or newer; Python 3.10-3.12 are the project's recommended versions.
- A webcam, compatible video file, or supported camera stream for non-demo use.
- Internet access on first use may be required to obtain machine-learning models. Phone detection
  additionally requires the optional dependencies in `requirements-phone.txt`.

## Quick start

From this directory, create and activate a virtual environment, then install the core
dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

Start the simulated room (no camera required):

```text
python run.py --demo
```

Open the dashboard URL printed in the terminal and sign in with a teacher name and the password
printed at startup. Use the dashboard's demo controls to generate sample incidents.

To try the supervised launcher, which creates the environment and installs core dependencies on
first start on Windows or macOS/Linux:

```text
start.bat
```

```bash
./start.sh
```

The launchers start `start.py`, which restarts the application after an unexpected exit. To
start an already-configured installation directly, use `python start.py` or `python run.py`.
Add the same options to `start.py` that you would pass to `run.py`.

## Using a camera

Install the optional phone/object detector if needed, then select a source:

```text
python -m pip install -r requirements-phone.txt
python run.py
python run.py --source 1
python run.py --source http://192.168.1.50:4747/video
python run.py --source exam.mp4
python run.py --no-phone
```

The default source is camera index `0`. `--source` accepts a camera index, stream URL, or video
file. Video files loop by default. `--no-phone` disables YOLO phone detection; the other
behavior rules can still run.

To let other devices on the local network access the dashboard, bind to all interfaces:

```text
python run.py --host 0.0.0.0
```

The startup banner prints the local address. By default, the dashboard binds to `127.0.0.1`
and is only available on the same computer.

Useful command-line options:

| Option | Purpose |
|---|---|
| `--demo` | Use a simulated room instead of a camera. |
| `--source SOURCE` | Camera index, stream URL, or video path. |
| `--second-source SOURCE` | Add an optional second camera. |
| `--room NAME` | Set the room name shown in the dashboard and incidents. |
| `--host HOST`, `--port PORT` | Set the dashboard bind address and port. |
| `--https` | Serve the dashboard over HTTPS using a self-signed certificate unless a certificate is configured. |
| `--no-phone` | Disable phone detection. |
| `--profile NAME` | Choose `laptop`, `edge`, `jetson`, or `gpu`. |
| `--role ROLE` | Choose `standalone`, `edge`, or `server`. |
| `-v`, `--verbose` | Enable verbose logging. |

Run `python run.py --help` for the current CLI help.

## Configuration

Copy `.env.example` to `.env` and edit the values you need. Real environment variables take
precedence over values in `.env`. Configuration fields in `examguard/config.py` can generally be
set as `EXAMGUARD_<FIELD_NAME_IN_UPPERCASE>`; `.env.example` lists commonly used settings.
Command-line options override corresponding configuration values.

Frequently used settings:

| Setting | Default | Description |
|---|---:|---|
| `EXAMGUARD_SOURCE` | `0` | Camera index, stream URL, or video path. |
| `EXAMGUARD_ROOM_NAME` | `Exam Room 1` | Room label. |
| `EXAMGUARD_PASSWORD` | generated | Dashboard password. If unset, a password is created and printed at startup, then saved under `data/`. |
| `EXAMGUARD_ROSTER_FILE` | unset | Optional `.csv` or `.xlsx` roster. |
| `EXAMGUARD_PROFILE` | `laptop` | Performance profile; profile values supply defaults only. |
| `EXAMGUARD_SECOND_SOURCE` | unset | Optional second camera source. |
| `EXAMGUARD_FACE_DETECTOR` | `auto` | Face detector selection: `auto`, `mediapipe`, or `staged`. |
| `EXAMGUARD_WRITE_MARKS` | `true` | Allow a teacher's confirmation to write a mark when a roster is configured. |
| `EXAMGUARD_HTTPS` | `false` | Enable HTTPS for the dashboard. |
| `EXAMGUARD_VAULT_KEY` | generated/managed by the vault | Optional externally managed key for evidence encryption. |

Any setting not shown in this table may be available in `examguard/config.py`; use the
`EXAMGUARD_` prefix and uppercase field name. For example, `yaw_threshold_deg` becomes
`EXAMGUARD_YAW_THRESHOLD_DEG`.

### Performance profiles

Choose a profile with `--profile` or `EXAMGUARD_PROFILE`. A profile supplies defaults; explicit
environment or command-line settings take precedence. The dashboard's System check page can
also save a profile choice; restart ExamGuard for the saved choice to take effect.

| Profile | Intended use | Notes |
|---|---|---|
| `laptop` | General laptop/PC, CPU | Default profile. |
| `edge` | Lower-power edge computer | Reduces frame size and processing rate and disables phone and hand detection. |
| `jetson` | NVIDIA Jetson-class computer | Enables CUDA for phone detection if the installed PyTorch build supports it. |
| `gpu` | More powerful NVIDIA GPU computer | Higher frame size/rate, CUDA phone detection, and staged face detection. |

Only the `laptop` profile is identified by the project as tested. Treat `edge`, `jetson`, and
`gpu` as starting configurations, not performance guarantees. Check System check and measure
performance and accuracy on the actual device.

## Detection and review workflow

The default rule thresholds are in `examguard/config.py`. Most can be adjusted through matching
`EXAMGUARD_*` environment variables.

| Rule | Default behavior |
|---|---|
| Sustained head turn | About 35 degrees from the student's baseline for 2 seconds. |
| Repeated glancing | Four shorter turns within 60 seconds. |
| Phone | A visible phone detected near a student for 1 second; YOLO dependencies are optional. |
| Looking down | A sustained downward head/eye cue beyond the student's baseline; default 3 seconds. |
| Repeated looking down | Four shorter downward looks within 60 seconds. |
| Eye gaze | Sustained sideways eye movement while the head faces forward; default 1.5 seconds. |
| Repeated eye gaze | Five shorter sideways glances within 60 seconds. |
| Talking | Repeated mouth movement among at least two nearby students for 3 seconds. |
| Possible passing | A hand appears near one student and then a nearby student within the configured interval. |

An alert for the same student and rule is rate-limited by a default 20-second cooldown. Rules
infer behavior from visual cues: for example, the system cannot see whether a student is reading
a hidden note, and the hand-passing rule does not identify an object.

An incident is initially pending. Teachers can review the clip and confirm or dismiss the alert;
dismissals can include a reason, and alerts can be ignored without being treated as a dismissal.
The dashboard also supports starting/finishing an exam, reporting a missed event, and viewing or
exporting a field report. Decisions and corrections are recorded in the audit trail.

### Evidence and retention

Evidence clips are approximately five seconds by default (3.5 seconds before and 1.5 seconds
after an alert). Evidence and thumbnails are stored encrypted in the data directory. Default
retention removes dismissed evidence after 1 hour, confirmed evidence after 30 days, and
unreviewed evidence after 14 days. The retention settings can be changed in `examguard/config.py`
or with corresponding `EXAMGUARD_*` variables.

## Optional features

### Student roster and marks

Without a roster, people are labelled by student number. Configure `EXAMGUARD_ROSTER_FILE` with
a `.csv` or `.xlsx` roster containing student names and identifiers; photos can be supplied for
face-match suggestions. See [roster/roster_template.csv](roster/roster_template.csv) and
[roster/photos/README.txt](roster/photos/README.txt) for the repository's roster examples.

Face matching is only a suggestion. Teachers should check the roster photo and the evidence
before associating a name with an incident. Face matching is biometric processing; obtain the
required institutional approval and consent before using it.

If spreadsheet marks are enabled, a teacher's **confirmation** can write the configured value
(`Fail` by default) to the roster's mark column. A backup is created before changing a file.
Test with a copy of the roster first, and close the spreadsheet in other applications before
using write-back. A mark is not written for an unidentified student. Configure
`EXAMGUARD_WRITE_MARKS`, `EXAMGUARD_MARKS_COLUMN`, and `EXAMGUARD_MARK_VALUE` as needed.

### Seat map

Open **Seat map** in the dashboard to create or adjust seat markers. You can fill an initial map
from detected faces or place seats manually, edit labels, and optionally assign roster students.
The map is stored under `data/seatmap.json` in standalone mode. In edge/server mode, edit it on
the server dashboard; it is synchronized to the edge.

Seat assignment is approximate and depends on the camera position and where students sit. Check
labels after creating a map and recreate it if the camera moves. Seat-based identity is a
suggestion, not proof of identity.

### Second camera

Pass `--second-source` or set `EXAMGUARD_SECOND_SOURCE` to add another view. Example:

```text
python run.py --source http://192.168.1.50:4747/video --second-source http://192.168.1.51:4747/video
```

The cameras share the incident list but each has its own view and student tracking. Two cameras
require additional processing capacity, and a person visible to both may generate separate
alerts. Use cameras to cover different areas and create a seat map for each view.

### Edge box and server

Standalone mode is the default. The optional split mode runs camera processing on an edge
computer and the teacher dashboard on a server computer, communicating through an MQTT broker.
Configure both with the same room name and `EXAMGUARD_MQTT_KEY`; generate a key with
`python scripts/make_key.py`. Broker examples and access-control starting points are in
`deploy/mosquitto.conf` and `deploy/acl`.

Start the edge process:

```text
python run.py --role edge
```

Start the dashboard/server process:

```text
python run.py --role server
```

Set the broker connection and credentials on both computers using the `EXAMGUARD_MQTT_*`
settings in `.env.example`. The edge stores undelivered incidents in an outbox; server-side
review and evidence retention remain in effect. The server live view is a periodic still image,
not a continuous video stream. The split path has automated tests using a test broker, but has
not been validated here with a real Mosquitto installation or two physical computers.

### Email and HTTPS

Optional Gmail notifications use an app password, not a regular Gmail password. Set
`EXAMGUARD_EMAIL_USER`, `EXAMGUARD_EMAIL_APP_PASSWORD`, and `EXAMGUARD_EMAIL_TO` in `.env`.
`EXAMGUARD_PUBLIC_URL` can provide the dashboard link included in the message. Student images
are not included unless `EXAMGUARD_EMAIL_SEND_THUMB=true`. Verify delivery with the intended
mail configuration before relying on it.

Use `--https` or `EXAMGUARD_HTTPS=true` to enable HTTPS. ExamGuard can create a self-signed
certificate, or you can configure `EXAMGUARD_TLS_CERT` and `EXAMGUARD_TLS_KEY`. A self-signed
certificate will produce a browser warning. HTTPS protects traffic on the network; it does not
make it safe to expose the dashboard to the public internet.

## Tuning and evaluation

Tune camera fit and normal head movement with:

```text
python scripts/tune.py
python scripts/tune.py --demo
python scripts/tune.py --pose-check
```

Replay recorded video and export alerts:

```text
python scripts/evaluate.py exam.mp4 --events-out alerts.csv
python scripts/evaluate.py exam.mp4 --labels labels.csv
python scripts/evaluate.py --demo
```

The labels CSV uses `start,end,kind`, where times are seconds or `m:ss`; supported kinds include
`head_turn`, `phone`, `looking_down`, `eye_gaze`, `talking`, and `passing`. Evaluation reports
recall, precision, delay, and false alarms against the labels. The demo evaluates the tool on
the simulator, not on classroom footage.

The guided self-test records a prompted video and creates labels:

```text
python scripts/selftest.py
python scripts/selftest.py --repeats 5 --talking --phone
```

It uses the configured camera by default. Review local recording and consent requirements before
capturing any person. `scripts/check_roster.py` can check a roster's photos before face matching.

### Face detector modes

Set `EXAMGUARD_FACE_DETECTOR` to `auto`, `mediapipe`, or `staged`. `staged` uses OpenCV YuNet
to find faces and MediaPipe to analyze face crops; tile count and detector score can be tuned
with `EXAMGUARD_DETECTOR_TILES` and `EXAMGUARD_DETECTOR_SCORE`. Small or distant faces still
lack detail for reliable eye and mouth cues, regardless of crop enlargement. The System check
page reports operational warnings such as low face size and available resources.

## Tests and project layout

Run the automated test suite from the project directory:

```text
pytest -q
```

Tests cover core rules and tracking, simulated processing, storage and web workflows, roster and
mark handling, evaluation, profiles, and split-mode components. Automated tests do not establish
accuracy or reliability in a real classroom or on untested hardware.

| Path | Purpose |
|---|---|
| `run.py` | Command-line application entry point. |
| `start.py`, `start.bat`, `start.sh` | Supervised launcher; platform scripts create an environment and install core requirements. |
| `examguard/` | Configuration, capture, vision, rules, storage, services, and split-mode implementation. |
| `examguard/web/` | Flask routes, dashboard templates, and static assets. |
| `scripts/` | Tuning, self-test, evaluation, roster check, and key-generation tools. |
| `tests/` | Automated tests. |
| `data/` | Runtime database, settings, seat map, and encrypted evidence. |

## Limitations and responsible use

- This software does not establish that misconduct occurred. Visual heuristics can miss behavior
  or flag ordinary actions such as writing, stretching, looking at a clock, or talking.
- Phone detection requires a visible phone and the optional YOLO dependencies. Small, obscured,
  or distant objects may not be detected.
- Face matching, the staged face detector, camera combinations, profiles other than `laptop`,
  external email delivery, and the MQTT split deployment require validation in their intended
  environment. Passing automated tests is not real-world validation.
- Performance and accuracy depend on camera placement, face size, lighting, hardware, and model
  availability. Tune and evaluate with representative footage before any operational use.
- Recording students and processing biometric data may be regulated. Follow applicable law and
  institutional policy, obtain required approvals and consent, inform students, secure access,
  and set retention appropriate to your requirements.
- Keep credentials and encryption keys out of source control. Do not expose the dashboard
  directly to the internet; restrict network access and use HTTPS when accessing it over a
  network.

