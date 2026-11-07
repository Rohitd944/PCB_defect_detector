import { useState } from "react";
import { toast } from "react-toastify";
import { FiSearch, FiRefreshCw, FiAlertCircle, FiLayers, FiInfo } from "react-icons/fi";
import api from "../services/api";
import ImageInputCard from "./ImageInputCard";
import WebcamCapture from "./WebcamCapture";
import HybridResult from "./HybridResult";
import "./HybridWorkspace.css";

/**
 * Reference-compare (hybrid) inspection workspace.
 *
 * Two independent image inputs:
 *   A. known-good reference PCB  -> uploaded once to /reference
 *   B. testing PCB               -> sent to /inspect/hybrid
 */
function HybridWorkspace() {
  const [refFile, setRefFile] = useState(null);
  const [refPreview, setRefPreview] = useState(null);
  const [refId, setRefId] = useState(null);
  const [refStatus, setRefStatus] = useState(null);
  const [refStatusTone, setRefStatusTone] = useState("ok");
  const [refUploading, setRefUploading] = useState(false);

  const [testFile, setTestFile] = useState(null);
  const [testPreview, setTestPreview] = useState(null);

  const [confidence, setConfidence] = useState(0.25);
  const [diffThreshold, setDiffThreshold] = useState(18);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [cameraTarget, setCameraTarget] = useState(null);

  const uploadReference = async (file) => {
    setRefUploading(true);
    setRefStatusTone("busy");
    setRefStatus("Uploading reference and extracting ORB features…");
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name || "reference PCB");

    try {
      const response = await api.post("/reference", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const data = response.data;
      setRefId(data.reference_id);
      setRefStatusTone("ok");
      setRefStatus(
        `Reference ready · ${data.width}×${data.height} · ${data.keypoints} ORB features`
      );
    } catch (err) {
      const msg = err?.response?.data?.detail || "Could not save the reference image.";
      setRefId(null);
      setRefStatusTone("error");
      setRefStatus(msg);
      toast.error(msg);
    } finally {
      setRefUploading(false);
    }
  };

  const handleReferenceFile = (file) => {
    if (!file) return;
    setRefFile(file);
    setRefPreview(URL.createObjectURL(file));
    uploadReference(file);
  };

  const handleReferenceRemove = () => {
    setRefFile(null);
    setRefPreview(null);
    setRefId(null);
    setRefStatus(null);
    setResult(null);
    setError(null);
  };

  const handleTestingFile = (file) => {
    if (!file) return;
    setTestFile(file);
    setTestPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleTestingRemove = () => {
    setTestFile(null);
    setTestPreview(null);
    setResult(null);
    setError(null);
  };

  const handleCapture = async (imageSrc) => {
    const target = cameraTarget;
    setCameraTarget(null);
    try {
      const response = await fetch(imageSrc);
      const blob = await response.blob();
      const name = target === "reference" ? "reference_capture.jpg" : "testing_capture.jpg";
      const file = new File([blob], name, { type: "image/jpeg" });
      if (target === "reference") handleReferenceFile(file);
      else handleTestingFile(file);
    } catch {
      toast.error("Could not read the captured image.");
    }
  };

  const handleAnalyze = async () => {
    if (!refId) {
      toast.error("Add a reference (known-good) PCB first.");
      return;
    }
    if (!testFile) {
      toast.error("Add the testing PCB image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", testFile);
    formData.append("reference_id", refId);
    formData.append("confidence", confidence);
    formData.append("iou", 0.4);
    formData.append("diff_threshold", diffThreshold);

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.post("/inspect/hybrid", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      const verdict = response.data?.verdict;
      if (verdict === "PASS") toast.success("PASS — testing board matches the reference.");
      else if (verdict === "FAIL") toast.error("FAIL — defects confirmed against the reference.");
      else toast.warning("REVIEW — manual inspection recommended.");
    } catch (err) {
      const msg = err?.response?.data?.detail || "Hybrid inspection failed. Check the backend.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    handleReferenceRemove();
    handleTestingRemove();
    setLoading(false);
    setError(null);
  };

  const canAnalyze = Boolean(refId) && Boolean(testFile) && !loading && !refUploading;

  return (
    <section className="hybrid" aria-label="Reference comparison workspace">
      <div className="hybrid__intro">
        <div className="hybrid__intro-icon">
          <FiLayers size={18} />
        </div>
        <div>
          <h2 className="hybrid__intro-title">Reference Comparison</h2>
          <p className="hybrid__intro-sub">
            Compare a testing PCB against a known-good board. YOLOv8 finds defects;
            the reference comparison verifies whether the board actually changed.
          </p>
        </div>
      </div>

      <div className="hybrid__inputs">
        <ImageInputCard
          title="Reference PCB"
          subtitle="Known-good board — the golden sample"
          badge="A · Reference"
          tone="reference"
          file={refFile}
          preview={refPreview}
          onFileChange={handleReferenceFile}
          onRemove={handleReferenceRemove}
          onOpenCamera={() => setCameraTarget("reference")}
          status={refStatus}
          statusTone={refStatusTone}
          disabled={loading || refUploading}
        />

        <div className="hybrid__link" aria-hidden="true">
          <span>vs</span>
        </div>

        <ImageInputCard
          title="Testing PCB"
          subtitle="Board under inspection — upload or camera"
          badge="B · Testing"
          tone="testing"
          file={testFile}
          preview={testPreview}
          onFileChange={handleTestingFile}
          onRemove={handleTestingRemove}
          onOpenCamera={() => setCameraTarget("testing")}
          disabled={loading}
        />
      </div>

      <div className="hybrid__controls">
        <div className="hybrid__control">
          <div className="hybrid__control-head">
            <span className="label-chip">YOLO Confidence</span>
            <span className="hybrid__control-val mono">{Math.round(confidence * 100)}%</span>
          </div>
          <input
            type="range"
            min="0.10"
            max="0.70"
            step="0.05"
            value={confidence}
            onChange={(event) => setConfidence(parseFloat(event.target.value))}
            className="hybrid__slider"
            aria-label="YOLO confidence threshold"
          />
        </div>

        <div className="hybrid__control">
          <div className="hybrid__control-head">
            <span className="label-chip">Difference Sensitivity</span>
            <span className="hybrid__control-val mono">{diffThreshold}</span>
          </div>
          <input
            type="range"
            min="12"
            max="40"
            step="1"
            value={diffThreshold}
            onChange={(event) => setDiffThreshold(parseInt(event.target.value, 10))}
            className="hybrid__slider"
            aria-label="Difference threshold"
          />
          <p className="hybrid__control-hint">
            Lower = more sensitive to tiny differences. Higher = ignores lighting and
            pixel-level noise.
          </p>
        </div>
      </div>

      <div className="hybrid__actions">
        <button
          type="button"
          className="ws-btn ws-btn--primary"
          onClick={handleAnalyze}
          disabled={!canAnalyze}
          aria-label="Run hybrid inspection"
        >
          {loading ? (
            <>
              <span className="ws-btn__spinner" aria-hidden="true" /> Comparing…
            </>
          ) : (
            <>
              <FiSearch size={16} /> Run Hybrid Inspection
            </>
          )}
        </button>

        {(refFile || testFile || result) && (
          <button type="button" className="ws-btn ws-btn--ghost" onClick={handleReset}>
            <FiRefreshCw size={16} /> Reset
          </button>
        )}
      </div>

      {!refId && !refUploading && (
        <p className="hybrid__tip">
          <FiInfo size={13} /> Add the known-good reference first — the testing image is
          aligned to it before differences are measured.
        </p>
      )}

      {loading && (
        <div className="hybrid__loading animate-fade-in">
          <span className="ws-btn__spinner" aria-hidden="true" />
          <div>
            <strong>Aligning and comparing</strong>
            <p className="mono">
              ORB features → RANSAC homography → exposure normalisation → SSIM → fusion
            </p>
          </div>
        </div>
      )}

      {!loading && error && (
        <div className="hybrid__error animate-fade-in">
          <FiAlertCircle size={26} className="hybrid__error-icon" />
          <h3 className="hybrid__error-title">Inspection Failed</h3>
          <p className="hybrid__error-msg">{error}</p>
        </div>
      )}

      {!loading && result && <HybridResult result={result} onReset={handleReset} />}

      {cameraTarget && (
        <WebcamCapture
          title={cameraTarget === "reference" ? "Reference PCB — Camera" : "Testing PCB — Camera"}
          onCapture={handleCapture}
          onClose={() => setCameraTarget(null)}
        />
      )}
    </section>
  );
}

export default HybridWorkspace;
