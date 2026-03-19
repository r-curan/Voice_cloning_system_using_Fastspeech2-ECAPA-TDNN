# Full graphical user interface for recording, embedding extraction, and synthesis
# Fixed: QMediaContent wrapping for setMedia() - no more TypeError on macOS
# Added: Pause / Resume button that works for both processed samples and synthesized audio
"""
this user interface is built using PyQt5 and provides the following features:
1. Record new speaker using microphone and generate embedding
2. List available speakers and their embeddings, with option to play processed sample
3. Synthesize speech from text using selected speaker embedding, with pitch/energy/duration controls
4. Playback controls for synthesized audio (play/pause/stop)
imp note: Make sure to run this with the virtual environment activated that has all dependencies installed, and ensure that the generate.py script is in the same directory or adjust the path accordingly.
---most importantly it can only vary energy,pitch and duration of synthesized audio by 0.1 so if varying those parameters
on the scale lower than that the generate.py must be used...or tuning of ui.py must be done
"""

import sys
import os
os.environ["QT_LOGGING_RULES"] = "*.warning=false;qt5ct.debug=false"

import torch
import torchaudio
import numpy as np
import threading
import time
import subprocess
import glob

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QListWidget,
    QMessageBox, QGroupBox, QStatusBar, QTextEdit, QComboBox,
    QSlider, QTabWidget
)
from PyQt5.QtCore import QUrl, Qt, pyqtSignal, QObject
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QFont

import sounddevice as sd

# ── embedding pipeline ────────────────────────────────────────
from model.pre_trained.ecapa_tdnn_loader import get_ECAPA_TDNN_MODEL, speaker_embedding_extractor
from audio.tools import trim_silence

def full_pipeline(audio_path, speaker_id, output_dir="embeddings/LibriTTS"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    out_wav = os.path.join(output_dir, f"{speaker_id}_processed.wav")
    out_pt  = os.path.join(output_dir, f"{speaker_id}.pt")

    audio, sr = torchaudio.load(audio_path)
    peak = torch.max(torch.abs(audio))
    if peak > 0:
        audio = audio / peak
    if audio.ndim > 1 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio = resampler(audio)
        sr = 16000
    audio = trim_silence(audio)
    audio = audio.squeeze(0).to(device).float()

    torchaudio.save(out_wav, audio.cpu().unsqueeze(0), sample_rate=sr)

    model = get_ECAPA_TDNN_MODEL(device=device)
    model.eval()
    with torch.no_grad():
        embedding = speaker_embedding_extractor(model, audio.unsqueeze(0))
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

    torch.save(embedding.cpu(), out_pt)
    return out_wav, out_pt, embedding


class RecorderSignals(QObject):
    update_time   = pyqtSignal(float)
    error_message = pyqtSignal(str)


class TTSApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Cloning - Dataset + Synthesis")
        self.resize(1050, 820)

        os.makedirs("createdataset", exist_ok=True)
        os.makedirs("embeddings/LibriTTS", exist_ok=True)

        self.recording     = False
        self.audio_chunks  = []
        self.sample_rate   = 44100
        self.start_t       = 0
        self.record_thread = None

        self.rec_signals = RecorderSignals()
        self.rec_signals.update_time.connect(self.update_rec_label)
        self.rec_signals.error_message.connect(self.show_rec_error)

        self.player = QMediaPlayer()
        self.last_synthesized_file = None

        self.init_ui()
        self.refresh_speakers()

    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        main_lay = QVBoxLayout(cw)

        tabs = QTabWidget()
        main_lay.addWidget(tabs)

        # ── Record Tab ───────────────────────────────────────────────
        tab_rec = QWidget()
        lay_rec = QVBoxLayout(tab_rec)

        g_rec = QGroupBox(" 1. Record new speaker (microphone) ")
        f_rec = QFormLayout()

        self.filename_edit = QLineEdit()
        self.filename_edit.setPlaceholderText("example: sandip_new_01")
        f_rec.addRow("Raw filename (saved in createdataset/):", self.filename_edit)

        btnrow = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start Recording")
        self.btn_start.clicked.connect(self.start_rec)
        self.btn_stop  = QPushButton("■ Stop & Generate Embedding")
        self.btn_stop.clicked.connect(self.stop_rec)
        self.btn_stop.setEnabled(False)
        btnrow.addWidget(self.btn_start)
        btnrow.addWidget(self.btn_stop)
        f_rec.addRow(btnrow)

        self.lbl_rec = QLabel("Ready")
        self.lbl_rec.setAlignment(Qt.AlignCenter)
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")
        f_rec.addRow(self.lbl_rec)

        g_rec.setLayout(f_rec)
        lay_rec.addWidget(g_rec)
        tabs.addTab(tab_rec, "Record")

        # ── Speakers Tab ─────────────────────────────────────────────
        tab_spk = QWidget()
        lay_spk = QVBoxLayout(tab_spk)

        g_spk = QGroupBox(" 2. Available Speakers / Embeddings ")
        v_spk = QVBoxLayout()

        self.lst_speakers = QListWidget()
        self.lst_speakers.itemDoubleClicked.connect(self.on_speaker_doubleclick)
        v_spk.addWidget(self.lst_speakers)

        h_btn = QHBoxLayout()
        btn_ref = QPushButton("↻ Refresh List")
        btn_ref.clicked.connect(self.refresh_speakers)
        btn_play = QPushButton("▶ Play Processed Sample")
        btn_play.clicked.connect(self.play_selected_sample)
        h_btn.addWidget(btn_ref)
        h_btn.addWidget(btn_play)
        v_spk.addLayout(h_btn)

        self.txt_details = QTextEdit(readOnly=True)
        self.txt_details.setMaximumHeight(160)
        v_spk.addWidget(QLabel("Embedding Details:"))
        v_spk.addWidget(self.txt_details)

        g_spk.setLayout(v_spk)
        lay_spk.addWidget(g_spk)
        tabs.addTab(tab_spk, "Speakers")

        # ── Synthesize Tab ───────────────────────────────────────────
        tab_syn = QWidget()
        lay_syn = QVBoxLayout(tab_syn)

        g_syn = QGroupBox(" 3. Text → Speech Synthesis ")
        f_syn = QFormLayout()

        self.cmb_speaker = QComboBox()
        f_syn.addRow("Select Voice:", self.cmb_speaker)

        self.txt_input = QTextEdit()
        self.txt_input.setPlaceholderText("Type your sentence here...")
        self.txt_input.setMaximumHeight(100)
        f_syn.addRow("Text to speak:", self.txt_input)

        h_ctrl = QHBoxLayout()
        for name, default in [("Pitch", 10), ("Energy", 10), ("Duration", 10)]:
            slider = QSlider(Qt.Horizontal)
            slider.setRange(5, 20)
            slider.setValue(default)
            label = QLabel(f"{default/10:.1f}")
            slider.valueChanged.connect(lambda v, lbl=label: lbl.setText(f"{v/10:.1f}"))
            h_ctrl.addWidget(QLabel(name + ":"))
            h_ctrl.addWidget(slider)
            h_ctrl.addWidget(label)
            setattr(self, f"slider_{name.lower()}", slider)
            setattr(self, f"lbl_{name.lower()}", label)
        f_syn.addRow(h_ctrl)

        btn_synth = QPushButton(" GENERATE AUDIO ")
        btn_synth.setStyleSheet("font-size:16px; padding:12px; background:#4CAF50; color:white;")
        btn_synth.clicked.connect(self.generate_audio)
        f_syn.addRow(btn_synth)

        self.log_syn = QTextEdit(readOnly=True)
        self.log_syn.setMaximumHeight(140)
        f_syn.addRow("Synthesis Log:", self.log_syn)

        # ── Added: Playback Controls ────────────────────────────────
        playback_row = QHBoxLayout()
        self.btn_play_pause = QPushButton("▶ Play / ⏸ Pause")
        self.btn_play_pause.clicked.connect(self.toggle_play_pause)
        playback_row.addWidget(self.btn_play_pause)

        btn_stop = QPushButton("⏹ Stop")
        btn_stop.clicked.connect(self.player.stop)
        playback_row.addWidget(btn_stop)

        btn_play_last = QPushButton("▶ Play Last Generated Audio")
        btn_play_last.clicked.connect(self.play_last_generated)
        playback_row.addWidget(btn_play_last)

        f_syn.addRow(playback_row)

        g_syn.setLayout(f_syn)
        lay_syn.addWidget(g_syn)
        tabs.addTab(tab_syn, "Synthesize")

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready - make sure venv is active")

    # ── New method: Toggle Play / Pause ─────────────────────────────
    def toggle_play_pause(self):
        state = self.player.state()
        if state == QMediaPlayer.PlayingState:
            self.player.pause()
            self.btn_play_pause.setText("▶ Resume")
        else:
            self.player.play()
            self.btn_play_pause.setText("⏸ Pause")

    # ── Recording Logic ──────────────────────────────────────────────
    def start_rec(self):
        if not self.filename_edit.text().strip():
            QMessageBox.warning(self, "Input required", "Please enter a filename first.")
            return

        self.audio_chunks = []
        self.recording = True
        self.start_t = time.time()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_rec.setText("Recording... 0.0 s")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#d32f2f;")

        self.record_thread = threading.Thread(target=self._rec_worker, daemon=True)
        self.record_thread.start()

    def _rec_worker(self):
        def cb(indata, frames, ti, status):
            if self.recording:
                self.audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(samplerate=self.sample_rate, channels=1,
                                dtype='float32', callback=cb):
                while self.recording:
                    elapsed = time.time() - self.start_t
                    self.rec_signals.update_time.emit(elapsed)
                    sd.sleep(150)
        except Exception as e:
            self.rec_signals.error_message.emit(str(e))

    def update_rec_label(self, sec):
        self.lbl_rec.setText(f"Recording... {sec:.1f} s")

    def show_rec_error(self, msg):
        QMessageBox.critical(self, "Recording Error", msg)
        self.reset_rec_ui()

    def stop_rec(self):
        self.recording = False
        self.lbl_rec.setText("Processing...")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")

        if self.record_thread and self.record_thread.is_alive():
            self.record_thread.join(timeout=5)

        if not self.audio_chunks:
            self.reset_rec_ui()
            return

        name = self.filename_edit.text().strip()
        raw_path = os.path.join("createdataset", f"{name}.wav")
        arr = np.concatenate(self.audio_chunks, axis=0)
        torchaudio.save(raw_path, torch.from_numpy(arr.T).float(), self.sample_rate)

        sid = self.next_speaker_id()

        try:
            pw, pt, _ = full_pipeline(raw_path, sid)
            QMessageBox.information(self, "Success",
                f"Speaker {sid} created successfully!\n\n"
                f"Raw audio     : {raw_path}\n"
                f"Processed wav : {pw}\n"
                f"Embedding     : {pt}")
            self.refresh_speakers()
        except Exception as e:
            QMessageBox.critical(self, "Pipeline Error", str(e))

        self.reset_rec_ui()

    def reset_rec_ui(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_rec.setText("Ready")
        self.lbl_rec.setStyleSheet("font-size:15px; font-weight:bold; color:#444;")
        self.audio_chunks = []

    def next_speaker_id(self):
        folder = "embeddings/LibriTTS"
        nums = []
        for f in os.listdir(folder):
            if f.endswith(".pt"):
                try:
                    nums.append(int(os.path.splitext(f)[0]))
                except:
                    pass
        return str(max(nums) + 1) if nums else "1"

    # ── Speakers List ────────────────────────────────────────────────
    def refresh_speakers(self):
        self.lst_speakers.clear()
        self.cmb_speaker.clear()

        folder = "embeddings/LibriTTS"
        pts = [f for f in os.listdir(folder) if f.endswith(".pt")]
        pts.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 999999)

        for f in pts:
            sid = os.path.splitext(f)[0]
            item_text = f"Speaker {sid}   •   {f}   •   {sid}_processed.wav"
            self.lst_speakers.addItem(item_text)
            self.cmb_speaker.addItem(sid)

    def on_speaker_doubleclick(self, item):
        sid = item.text().split()[1]
        self.show_speaker_info(sid)

    def show_speaker_info(self, sid):
        path = os.path.join("embeddings/LibriTTS", f"{sid}.pt")
        if not os.path.exists(path):
            return

        emb = torch.load(path)
        info = f"<b>Speaker ID:</b> {sid}<br>" \
               f"<b>Embedding shape:</b> {emb.shape}<br>" \
               f"<b>min / max:</b> {emb.min():.6f} / {emb.max():.6f}<br>" \
               f"<b>L2 norm:</b> {emb.norm():.6f}<br><br>" \
               f"First 20 values: {emb[0][:20].tolist()}"
        self.txt_details.setHtml(info)

        self.play_processed(sid)

    def play_processed(self, sid):
        p = os.path.join("embeddings/LibriTTS", f"{sid}_processed.wav")
        if os.path.exists(p):
            url = QUrl.fromLocalFile(os.path.abspath(p))
            self.player.setMedia(QMediaContent(url))
            self.player.play()
            self.btn_play_pause.setText("⏸ Pause")  # reset button text
        else:
            print(f"Processed file not found: {p}")

    def play_selected_sample(self):
        item = self.lst_speakers.currentItem()
        if item:
            sid = item.text().split()[1]
            self.play_processed(sid)

    # ── Synthesis ────────────────────────────────────────────────────
    def generate_audio(self):
        speaker = self.cmb_speaker.currentText()
        text = self.txt_input.toPlainText().strip()
        if not speaker or not text:
            QMessageBox.warning(self, "Missing input", "Select a speaker and enter text.")
            return

        p_val = self.slider_pitch.value()    / 10.0
        e_val = self.slider_energy.value()   / 10.0
        d_val = self.slider_duration.value() / 10.0

        self.log_syn.clear()
        self.log_syn.append("Starting synthesis...")

        cmd = [
            sys.executable, "generate.py",
            "--restore_step", "450000",
            "--mode", "single",
            "--text", text,
            "--speaker_emb", f"embeddings/LibriTTS/{speaker}.pt",
            "-p", "config/LibriTTS/preprocess.yaml",
            "-m", "config/LibriTTS/model.yaml",
            "-t", "config/LibriTTS/train.yaml",
            "--pitch_control",   f"{p_val:.2f}",
            "--energy_control",  f"{e_val:.2f}",
            "--duration_control",f"{d_val:.2f}"
        ]

        threading.Thread(target=self._run_synth, args=(cmd,), daemon=True).start()

    def _run_synth(self, cmd):
        out_dir = "output/result/LibriTTS"

        before = set(glob.glob(os.path.join(out_dir, "*.wav"))) if os.path.exists(out_dir) else set()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
            log_text = result.stdout + "\n" + result.stderr
            self.log_syn.append(log_text)

            time.sleep(1.0)
            after = set(glob.glob(os.path.join(out_dir, "*.wav"))) if os.path.exists(out_dir) else set()

            new_files = after - before
            if new_files:
                newest = max(new_files, key=os.path.getmtime)
                self.last_synthesized_file = newest
                self.log_syn.append(f"\n→ Detected new file: {os.path.basename(newest)}")
                self.log_syn.append(f"   Full path: {newest}")
            else:
                self.log_syn.append("\nNo new .wav detected in output/result/LibriTTS/")

            if result.returncode == 0:
                self.log_syn.append("\nGeneration completed successfully ✓")
            else:
                self.log_syn.append("\nGeneration failed ✗  (check log)")
        except Exception as ex:
            self.log_syn.append(f"\nException: {ex}")

    def play_last_generated(self):
        if not self.last_synthesized_file or not os.path.exists(self.last_synthesized_file):
            QMessageBox.information(self, "No file", "No recent synthesis found or file missing.")
            return

        url = QUrl.fromLocalFile(os.path.abspath(self.last_synthesized_file))
        self.player.setMedia(QMediaContent(url))
        self.player.play()
        self.btn_play_pause.setText("⏸ Pause")  # reset button text
        self.status.showMessage(f"Playing: {os.path.basename(self.last_synthesized_file)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TTSApp()
    window.show()
    sys.exit(app.exec_())