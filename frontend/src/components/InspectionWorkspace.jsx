import { useState } from "react";
import { toast } from "react-toastify";
import { FiUploadCloud, FiCamera, FiSearch, FiRefreshCw, FiDownload, FiAlertCircle } from "react-icons/fi";
import api from "../services/api";
import UploadZone from "./UploadZone";
import WebcamCapture from "./WebcamCapture";
import PCBViewer from "./PCBViewer";
import DetectionList from "./DetectionList";
import InspectionSummary from "./InspectionSummary";
import AnalysisState from "./AnalysisState";
import ReportPanel from "./ReportPanel";
import "./InspectionWorkspace.css";

function InspectionWorkspace() {
  const [file, setFile]               = useState(null);
  const [preview, setPreview]         = useState(null);
  const [loading, setLoading]         = useState(false);
  const [result, setResult]           = useState(null);
  const [error, setError]             = useState(null);
  const [showCamera, setShowCamera]   = useState(false);
  const [selectedDet, setSelectedDet] = useState(null);
  const [inputMode, setInputMode]     = useState("upload"); // "upload" | "camera"
  const [confidence, setConfidence]   = useState(0.25);

  const handleFileSelect = (f) => {
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setSelectedDet(null);
    setInputMode("upload");
  };

  const handleRemoveFile = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setSelectedDet(null);
  };

  const handleCapture = async (imageSrc) => {
    setPreview(imageSrc);
    const res  = await fetch(imageSrc);
    const blob = await res.blob();
    const f    = new File([blob], "captured_pcb.jpg", { type: "image/jpeg" });
    setFile(f);
    setResult(null);
    setError(null);
    setSelectedDet(null);
    setShowCamera(false);
    setInputMode("camera");
  };

  const handleAnalyze = async () => {
    if (!file) {
      toast.error("Select or capture a PCB image first.");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);
    formData.append("confidence", confidence);

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedDet(null);

    try {
      const response = await api.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      toast.success("PCB analyzed successfully.");
    } catch (err) {
      const msg = err?.response?.data?.detail || "Prediction failed. Check backend connection.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setSelectedDet(null);
    setInputMode("upload");
  };

  return (
    <section className="workspace" aria-label="Inspection workspace">

      {/* ── Source selector ── */}
      <div className="workspace__source">
        <div className="workspace__source-header">
          <span className="label-chip">Source</span>
          <div className="workspace__source-tabs">
            <button
              className={`workspace__source-tab ${inputMode === "upload" ? "active" : ""}`}
              onClick={() => setInputMode("upload")}
            >
              <FiUploadCloud size={14} /> Upload
            </button>
            <button
              className={`workspace__source-tab ${inputMode === "camera" ? "active" : ""}`}
              onClick={() => { setInputMode("camera"); setShowCamera(true); }}
            >
              <FiCamera size={14} /> Camera
            </button>
          </div>
        </div>

        <UploadZone
          file={file}
          onFileChange={handleFileSelect}
          onRemove={handleRemoveFile}
        />

        {/* ── Confidence Filter / Sensitivity ── */}
        <div className="workspace__sensitivity">
          <div className="workspace__sensitivity-header">
            <span className="label-chip">Confidence Filter</span>
            <span className="workspace__sensitivity-val mono">
              {Math.round(confidence * 100)}%
            </span>
          </div>
          <div className="workspace__slider-wrap">
            <input
              type="range"
              min="0.10"
              max="0.70"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
              className="workspace__slider"
              aria-label="Confidence threshold"
            />
          </div>
          <div className="workspace__presets">
            <button
              type="button"
              className={`ws-preset-btn ${confidence === 0.15 ? "active" : ""}`}
              onClick={() => setConfidence(0.15)}
            >
              Sensitive (15%)
            </button>
            <button
              type="button"
              className={`ws-preset-btn ${confidence === 0.25 ? "active" : ""}`}
              onClick={() => setConfidence(0.25)}
            >
              Balanced (25%)
            </button>
            <button
              type="button"
              className={`ws-preset-btn ${confidence === 0.45 ? "active" : ""}`}
              onClick={() => setConfidence(0.45)}
            >
              Strict (45%)
            </button>
          </div>
          <p className="workspace__sensitivity-hint">
            {confidence >= 0.40
              ? "Strict (40%+): Only very strong defect signals. Best for grayscale/scanned PCB images."
              : confidence <= 0.18
              ? "Sensitive (10-18%): May catch faint defects but also flags normal solder pads as false positives."
              : "Balanced (20-35%): Recommended — filters out false alarms on normal pads while catching real defects."}
          </p>
        </div>

        <div className="workspace__actions">
          <button
            className="ws-btn ws-btn--primary"
            onClick={handleAnalyze}
            disabled={!file || loading}
            aria-label="Analyze PCB"
          >
            {loading ? (
              <><span className="ws-btn__spinner" aria-hidden="true" /> Analyzing...</>
            ) : (
              <><FiSearch size={16} /> Analyze PCB</>
            )}
          </button>

          {(file || result) && (
            <button className="ws-btn ws-btn--ghost" onClick={handleReset} aria-label="Reset">
              <FiRefreshCw size={16} /> Reset
            </button>
          )}
        </div>
      </div>

      {/* ── Main workspace ── */}
      <div className="workspace__main">

        {/* Loading state */}
        {loading && <AnalysisState preview={preview} />}

        {/* Error state */}
        {!loading && error && (
          <div className="workspace__error animate-fade-in">
            <FiAlertCircle size={28} className="workspace__error-icon" />
            <h3 className="workspace__error-title">Inspection Failed</h3>
            <p className="workspace__error-msg">{error}</p>
            <button className="ws-btn ws-btn--ghost" onClick={handleAnalyze}>
              Try Again
            </button>
          </div>
        )}

        {/* Ready / results */}
        {!loading && !error && (
          <>
            {/* Viewer + detection panel */}
            <div className="workspace__viewer-row">
              <div className="workspace__viewer-col">
                {result && (
                  <div className="workspace__insp-id mono">
                    INSPECTION — {result.analysis.inspection_id}
                  </div>
                )}
                <PCBViewer
                  original={preview}
                  resultUrl={result?.result_image_url}
                  detections={result?.detections}
                  selectedIndex={selectedDet}
                  onSelectDetection={setSelectedDet}
                />
              </div>

              {result && (
                <div className="workspace__det-col animate-fade-in">
                  <span className="label-chip" style={{ marginBottom: "var(--space-3)" }}>
                    Detections
                  </span>
                  <DetectionList
                    detections={result.detections}
                    selectedIndex={selectedDet}
                    onSelect={setSelectedDet}
                  />
                </div>
              )}
            </div>

            {/* Results bottom section */}
            {result && (
              <div className="workspace__results animate-fade-in">
                <div className="divider" />

                <span className="label-chip">Inspection Summary</span>
                <div style={{ marginTop: "var(--space-4)" }}>
                  <InspectionSummary result={result} />
                </div>

                <div className="divider" />

                <div className="workspace__report-row">
                  <div className="workspace__report-col">
                    <span className="label-chip" style={{ marginBottom: "var(--space-4)" }}>
                      Report
                    </span>
                    <ReportPanel
                      analysis={result.analysis}
                      detections={result.detections}
                      imageUrl={result.result_image_url}
                    />
                  </div>
                </div>

                <div className="workspace__final-actions">
                  <a
                    href={result.result_image_url}
                    download
                    className="ws-btn ws-btn--ghost"
                    aria-label="Download annotated result"
                  >
                    <FiDownload size={16} /> Download Annotated Image
                  </a>
                  <button className="ws-btn ws-btn--ghost" onClick={handleReset}>
                    <FiRefreshCw size={16} /> New Inspection
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {showCamera && (
        <WebcamCapture
          onCapture={handleCapture}
          onClose={() => setShowCamera(false)}
        />
      )}
    </section>
  );
}

export default InspectionWorkspace;
