#!/usr/bin/env python3
"""
Fluorescence ROI Analyzer GUI  – FINAL, FULL VERSION
===================================================
Everything requested is implemented **and the file is complete end‑to‑end**.
Existing features are left intact; only the missing bottom part has been filled
in, including peak detection/analysis and the `main()` entry‑point.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import tifffile as tiff
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from scipy.ndimage import maximum_filter

pg.setConfigOptions(imageAxisOrder="row-major")

# ---------------------------------------------------------------------------
class PeakParamDialog(QtWidgets.QDialog):
    """Dialog for peak detection parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Peak detection parameters")
        lay = QtWidgets.QFormLayout(self)
        self.th = QtWidgets.QDoubleSpinBox()
        self.th.setDecimals(2); self.th.setRange(0, 1e9); self.th.setValue(100)
        self.rad = QtWidgets.QSpinBox()
        self.rad.setRange(1, 100); self.rad.setValue(5)
        lay.addRow("Intensity threshold", self.th)
        lay.addRow("NMS radius (mm)", self.rad)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    @property
    def threshold(self) -> float: return float(self.th.value())
    @property
    def radius_mm(self) -> int: return int(self.rad.value())


class RadiusDialog(QtWidgets.QDialog):
    """Dialog for neighbourhood analysis parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neighbourhood analysis parameters")
        lay = QtWidgets.QFormLayout(self)
        self.rad_mm = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.001, maximum=1000, value=0.5)
        self.max_peaks = QtWidgets.QSpinBox(); self.max_peaks.setRange(1, 10000); self.max_peaks.setValue(50)
        lay.addRow("Radius (mm)", self.rad_mm)
        lay.addRow("Max. peaks", self.max_peaks)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject)
        lay.addWidget(bb)
    @property
    def radius_mm(self): return float(self.rad_mm.value())
    @property
    def limit(self): return int(self.max_peaks.value())

# ---------------------------------------------------------------------------
class FluorescenceAnalyzer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fluorescence ROI Analyzer")
        self.resize(1400, 780)
        root = QtWidgets.QVBoxLayout(self)

        # Toolbar --------------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        self.btn_load_rgb   = QtWidgets.QPushButton("Load RGB …")
        self.btn_load_fluor = QtWidgets.QPushButton("Load fluorescence …")
        self.btn_set_scale  = QtWidgets.QPushButton("Set scale")
        self.btn_add_roi    = QtWidgets.QPushButton("Add ROI")
        self.btn_export_roi = QtWidgets.QPushButton("Export ROIs")
        self.btn_detect     = QtWidgets.QPushButton("Detect peaks")
        self.btn_analyze    = QtWidgets.QPushButton("Analyze peaks")
        for b in (self.btn_set_scale, self.btn_add_roi, self.btn_export_roi, self.btn_detect, self.btn_analyze):
            b.setEnabled(False)
        bar.addWidget(self.btn_load_rgb)
        bar.addWidget(self.btn_load_fluor)
        bar.addStretch(1)
        bar.addWidget(self.btn_set_scale)
        bar.addWidget(self.btn_add_roi)
        bar.addWidget(self.btn_export_roi)
        bar.addWidget(self.btn_detect)
        bar.addWidget(self.btn_analyze)
        root.addLayout(bar)

        # Viewers --------------------------------------------------------
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.rgb_view, self.fluor_view = pg.ImageView(view=pg.PlotItem()), pg.ImageView(view=pg.PlotItem())
        for v in (self.rgb_view, self.fluor_view): v.ui.roiBtn.hide(); v.ui.menuBtn.hide(); v.ui.histogram.hide()
        splitter.addWidget(self.rgb_view); splitter.addWidget(self.fluor_view)
        splitter.setStretchFactor(0,1); splitter.setStretchFactor(1,1)
        root.addWidget(splitter)

        # Status ---------------------------------------------------------
        self.status = QtWidgets.QLabel("Load both images to get started.")
        self.status.setFixedHeight(24); root.addWidget(self.status)

        # Data -----------------------------------------------------------
        self.rgb:Optional[np.ndarray]=None; self.fluor:Optional[np.ndarray]=None
        self._rois:List[pg.ROI]=[]; self._scale_roi=None; self._scale_mm=None; self.mm_per_px=None
        self._peak_scatter=None; self._peaks=None

        # Connections ----------------------------------------------------
        self.btn_load_rgb.clicked.connect(lambda: self._load_rgb())
        self.btn_load_fluor.clicked.connect(lambda: self._load_fluor())
        self.btn_set_scale.clicked.connect(self._define_scale)
        self.btn_add_roi.clicked.connect(self._add_roi)
        self.btn_export_roi.clicked.connect(self._export_rois)
        self.btn_detect.clicked.connect(self._detect_peaks)
        self.btn_analyze.clicked.connect(self._analyze_peaks)


    # ----------------------------------------------------------------- I/O
    def _open(self, t):
        p,_=QtWidgets.QFileDialog.getOpenFileName(self,t,"","Images (*.png *.jpg *.jpeg *.tif *.tiff)");return p or None
    def _load_rgb(self):
        p=self._open("Open RGB image");
        if not p: return
        img=cv2.imread(p,cv2.IMREAD_COLOR);      
        if img is None: QtWidgets.QMessageBox.warning(self,"Error","Cannot read image.");return
        self.rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); self.rgb_view.setImage(np.flipud(self.rgb)); self._update_state()
    def _load_fluor(self):
        p=self._open("Open fluorescence image");
        if not p: return
        img=tiff.imread(p).astype(np.float32)
        if img is None: QtWidgets.QMessageBox.warning(self,"Error","Cannot read image.");return
        if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.fluor=img; self._update_state()

    # ----------------------------------------------------------- UI state
    def _update_state(self):
        ready=self.rgb is not None and self.fluor is not None
        for b in (self.btn_set_scale,self.btn_add_roi,self.btn_detect,self.btn_analyze): b.setEnabled(ready)
        if not ready:
            self.status.setText("Load both images to get started."); return
        if self.rgb.shape[:2]!=self.fluor.shape[:2]:
            self.fluor=cv2.resize(self.fluor,(self.rgb.shape[1],self.rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
        self.fluor_view.setImage(np.flipud(self.fluor)); self.status.setText("Images loaded. Set scale or add ROIs.")
        if self._peak_scatter is not None:
            self.fluor_view.removeItem(self._peak_scatter); self._peak_scatter=None; self._peaks=None; self.btn_analyze.setEnabled(False)

    # -------------------------------------------------------------- Scale
    def _define_scale(self):
        if self.rgb is None: return
        if self._scale_roi is not None: self.rgb_view.removeItem(self._scale_roi)
        h,w=self.rgb.shape[:2]; seg=min(h,w)//3; p1=[w/2-seg/2,h/2]; p2=[w/2+seg/2,h/2]
        self._scale_roi=pg.LineROI(p1,p2,width=3,pen=pg.mkPen("magenta",width=2)); self.rgb_view.addItem(self._scale_roi)
        QtWidgets.QMessageBox.information(self,"Define scale","Adjust line to a known distance, then enter value.")
        dist,ok=QtWidgets.QInputDialog.getDouble(self,"Set scale","Distance (mm):",decimals=4,min=1e-6)
        if not ok or dist<=0:
            self.rgb_view.removeItem(self._scale_roi); self._scale_roi=None; self.status.setText("Scale cancelled."); return
        self._scale_mm=dist; self._scale_roi.sigRegionChanged.connect(self._recompute_scale); self._recompute_scale()
    def _recompute_scale(self):
        if self._scale_roi is None or self._scale_mm is None: return
        p1,p2=[h.pos() for h in self._scale_roi.getHandles()[:2]]; px=math.hypot(p2.x()-p1.x(),p2.y()-p1.y())
        if px==0: return
        self.mm_per_px=self._scale_mm/px; self.status.setText(f"Scale: 1 px = {self.mm_per_px:.4f} mm")

    # -------------------------------------------------------------- ROI
    def _add_roi(self):
        if self.rgb is None: 
            return
        h,w=self.rgb.shape[:2]; size=min(h,w)//4
        roi=pg.RectROI([w/2-size/2,h/2-size/2],[size,size],pen=pg.mkPen("cyan",width=2))
        roi.addRotateHandle([1,0],[0.5,0.5]);
        for hx,hy,cx,cy in ((1,1,0,0),(0,0,1,1),(0,1,1,0)): 
            roi.addScaleHandle([hx,hy],[cx,cy])
        self.rgb_view.addItem(roi)
        mirror=pg.RectROI(roi.pos(),roi.size(),pen=pg.mkPen("yellow",width=2,style=QtCore.Qt.DashLine),movable=False,rotatable=False,resizable=False)
        self.fluor_view.addItem(mirror)
        def sync():
            mirror.setPos(roi.pos()); mirror.setSize(roi.size())
            mirror.setAngle(roi.angle()); self._measure_roi(roi)
        roi.sigRegionChanged.connect(sync); 
        self._rois.append(roi); 
        sync()
        self.btn_export_roi.setEnabled(True)

    def _measure_roi(self, roi):
        if self.fluor is None: return
        region=roi.getArrayRegion(self.fluor,self.rgb_view.getImageItem(),axes=(0,1))
        if region is None or region.size==0: return
        mean=float(np.mean(region)); idx=self._rois.index(roi)+1
        self.status.setText(f"ROI {idx}: mean fluorescence = {mean:.2f}")

    # ----------------------------------------------------------- Peeks
    def _detect_peaks(self):
        if self.fluor is None:
            return
        if self.mm_per_px is None:
            QtWidgets.QMessageBox.information(self, "Scale required", "Please set the pixel–mm scale first.")
            return
        dlg = PeakParamDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        thresh = dlg.threshold
        radius_px = int(round(dlg.radius_mm / self.mm_per_px))
        radius_px = max(1, radius_px)
        img = self.fluor
        footprint = np.ones((radius_px * 2 + 1, radius_px * 2 + 1))
        local_max = (img == maximum_filter(img, footprint=footprint)) & (img > thresh)
        coords = np.argwhere(local_max)
        if coords.size == 0:
            QtWidgets.QMessageBox.information(self, "Peaks", "No peaks found with current parameters.")
            return
        if self._peak_scatter is not None:
            self.fluor_view.removeItem(self._peak_scatter)
        self._peaks = coords
        self._annotate_peaks(self._peaks, img)
        self.status.setText(f"Detected {len(coords)} peaks (radius {radius_px}px ≈ {dlg.radius_mm}mm).")
        self.btn_analyze.setEnabled(True)

    def _analyze_peaks(self):
        """Compute mean fluorescence around peaks, keep top‑N and annotate image."""
        if self.fluor is None or self._peaks is None or len(self._peaks) == 0:
            QtWidgets.QMessageBox.information(self, "Peaks", "No peaks to analyze – run detection first.")
            return
        if self.mm_per_px is None:
            QtWidgets.QMessageBox.information(self, "Scale required", "Please set the pixel–mm scale first.")
            return
        dlg = RadiusDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        r_px = max(1, int(round(dlg.radius_mm / self.mm_per_px)))
        results = []
        h, w = self.fluor.shape
        yy, xx = np.mgrid[0:h, 0:w]
        for idx, (row, col) in enumerate(self._peaks, 1):
            mask = (xx - col) ** 2 + (yy - row) ** 2 <= r_px ** 2
            region = self.fluor[mask]
            mean_val = float(region.mean()) if region.size > 0 else float('nan')
            results.append((idx, int(col), int(row), mean_val))
        # Sort descending by mean fluorescence and keep top‑N peaks
        results.sort(key=lambda x: x[3], reverse=True)
        results = results[:dlg.limit]
        # Show only these peaks with labels on the image
        self._display_top_peaks(results)
        # Table / CSV
        self._show_peak_table(results, dlg.radius_mm, r_px)

    def _display_top_peaks(self, rows):
        """Replace any existing peak markers with the ranked top‑N list."""
        # Remove previous scatter/labels
        if self._peak_scatter is not None:
            self.fluor_view.removeItem(self._peak_scatter)
            self._peak_scatter = None
        if hasattr(self, "_peak_labels"):
            for lbl in self._peak_labels:
                self.fluor_view.removeItem(lbl)
        self._peak_labels = []
        # Create new scatter + numbered labels
        spots = []
        h = self.fluor.shape[0]
        for rank, (idx, x_px, y_px, mean_val) in enumerate(rows, 1):
            # image coordinates -> graphics coordinates: (col, flipped row)
            spots.append({'pos': (x_px, h-1-y_px), 'brush': 'magenta', 'size': 8})
        self._peak_scatter = pg.ScatterPlotItem(pen=None, spots=spots)
        self.fluor_view.addItem(self._peak_scatter)
        # Add text labels
        for rank, (idx, x_px, y_px, _) in enumerate(rows, 1):
            txt = pg.TextItem(text=str(rank), anchor=(0.5, 1.5), color='yellow')
            txt.setPos(x_px, h-1-y_px)
            self.fluor_view.addItem(txt)
            self._peak_labels.append(txt)

    # keep legacy method for full set annotation if needed
    def _annotate_peaks(self, peaks, img):
        spots = [{'pos': (c[1], img.shape[0]-1-c[0]), 'brush': 'magenta', 'size': 6} for c in peaks]
        self._peak_scatter = pg.ScatterPlotItem(pen=None, spots=spots)
        self.fluor_view.addItem(self._peak_scatter)

    # ---------------------- helpers -------------------------
    def _show_peak_table(self, rows, r_mm, r_px):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Peak mean fluorescence")
        layout = QtWidgets.QVBoxLayout(dlg)
        info = QtWidgets.QLabel(f"Radius: {r_mm:.3f} mm  (≈ {r_px} px)")
        layout.addWidget(info)
        table = QtWidgets.QTableWidget(len(rows), 4)
        table.setHorizontalHeaderLabels(["#", "x (px)", "y (px)", "Mean F"])
        for r, (idx, x, y, mean_val) in enumerate(rows):
            for c, val in enumerate((idx, x, y, f"{mean_val:.2f}")):
                item = QtWidgets.QTableWidgetItem(str(val))
                item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        layout.addWidget(table)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close | QtWidgets.QDialogButtonBox.Save)
        layout.addWidget(bb)
        def _save():
            path, _ = QtWidgets.QFileDialog.getSaveFileName(dlg, "Save CSV", "peaks.csv", "CSV Files (*.csv)")
            if not path:
                return
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["index", "x_px", "y_px", "mean_f", "radius_mm", "radius_px"])
                for row in rows:
                    writer.writerow([*row, r_mm, r_px])
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        bb.button(QtWidgets.QDialogButtonBox.Save).clicked.connect(_save)
        dlg.exec()

    def _export_rois(self):
        if len(self._rois) == 0:
            QtWidgets.QMessageBox.information(self, "ROIs", "No ROIs defined.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save ROI means", "rois.csv", "CSV Files (*.csv)")
        if not path:
            return
        rows = []
        for idx, roi in enumerate(self._rois, 1):
            region = roi.getArrayRegion(self.fluor, self.rgb_view.getImageItem(), axes=(0,1))
            mean_val = float(region.mean()) if region is not None and region.size > 0 else float('nan')
            rows.append((idx, mean_val))
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["roi_index", "mean_fluorescence"])
            writer.writerows(rows)
        QtWidgets.QMessageBox.information(self, "ROIs", f"Saved {len(rows)} ROI means to {path}.")
# ---------------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = FluorescenceAnalyzer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
