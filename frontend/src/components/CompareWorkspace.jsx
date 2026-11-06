import { useState } from "react";
import { toast } from "react-toastify";
import { FiSearch, FiRefreshCw, FiDownload, FiAlertCircle, FiCheckCircle, FiInfo } from "react-icons/fi";
import api from "../services/api";
import UploadZone from "./UploadZone";
import PCBViewer from "./PCBViewer";
import DetectionList from "./DetectionList";
import InspectionSummary from "./InspectionSummary";
import ReportPanel from "./ReportPanel";
import AnalysisState from "./AnalysisState";
import "./CompareWorkspace.css";

function CompareWorkspace() {
  const [refFile,    setRefFile]    = useState(null);
  const [refPreview, setRefPreview] = useState(null);
  const [testFile,   setTestFile]   = useState(null);
  const [testPreview,setTestPreview]= useState(null);

  const [loading,    setLoading]    = useState(false);
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);
  const [selectedDet,setSelectedDet]= useState(null);

  // Diff sensitivity threshold (10 = very sensitive, 60 = strict)
  const [diffThreshold, setDiffThreshold] = useState(30);

  const handleRefSelect = (f) => {
    setRefFile(f);
    setRefPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setSelectedDet(null);
  };

  const handleTestSelect = (f) => {
    setTestFile(f);
    setTestPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
    setSelectedDet(null);
  };

  const handleCompare = async () => {
    if (!refFile)  { toast.error("Reference (good) PCB image select karo.");  return; }
    if (!testFile) { toast.error("Test PCB image select karo."); return; }

    const formData = new FormData();
    formData.append("reference",      refFile);
    formData.append("test",           testFile);
    formData.append("diff_threshold", diffThreshold);

    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedDet(null);

    try {
      const response = await api.post("/compare", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      const n = response.data.total_detections;
      toast.success(n === 0 ? "No defects found — PCB looks good!" : `${n} defect region(s) detected.`);
    } catch (err) {
      const msg = err?.response?.data?.detail || "Comparison failed. Check backend connection.";
      setError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setRefFile(null);  setRefPreview(null);
    setTestFile(null); setTestPreview(null);
    setResult(null);   setError(null);
    setLoading(false); setSelectedDet(null);
  };

  const canCompare = refFile && testFile && !loading;

  return (
    <section className="cmp-workspace" aria-label="Reference comparison workspace">

      {/* ── Info banner ── */}
      <div className="cmp-info-banner">
        <FiInfo size={16} className="cmp-info-banner__icon" />
        <p>
          <strong>Reference Comparison Mode</strong> — Upload a known-good PCB as reference,
          then upload the board you want to inspect. The system will align both images and
          highlight every difference as a potential defect.
        </p>
      </div>

      {/* ── Upload panels ── */}
      <div className="cmp-upload-row">

        {/* Reference image */}
        <div className="cmp-upload-panel">
          <div className="cmp-upload-panel__header">
            <span className="label-chip">① Reference PCB</span>
            <span className="cmp-badge cmp-badge--good">Good Board</span>
          </div>
          <p className="cmp-upload-panel__hint">
            Upload a defect-free PCB image. This is the gold standard that the test board will be compared against.
          </p>
          <UploadZone
            file={refFile}
            onFileChange={handleRefSelect}
            onRemove={() => { setRefFile(null); setRefPreview(null); }}
          />
          {refPreview && (
            <div className="cmp-thumb-wrap">
              <img src={refPreview} alt="Reference PCB preview" className="cmp-thumb" />
            </div>
          )}
        </div>

        {/* VS divider */}
        <div className="cmp-vs" aria-hidden="true">VS</div>

        {/* Test image */}
        <div className="cmp-upload-panel">
          <div className="cmp-upload-panel__header">
            <span className="label-chip">② Test PCB</span>
            <span className="cmp-badge cmp-badge--test">Board Under Inspection</span>
          </div>
          <p className="cmp-upload-panel__hint">
            Upload the PCB board you want to inspect. The system will find all differences compared to the reference.
          </p>
          <UploadZone
            file={testFile}
            onFileChange={handleTestSelect}
            onRemove={() => { setTestFile(null); setTestPreview(null); }}
          />
          {testPreview && (
            <div className="cmp-thumb-wrap">
              <img src={testPreview} alt="Test PCB preview" className="cmp-thumb" />
            </div>
          )}
        </div>
      </div>

      {/* ── Sensitivity control ── */}
      <div className="cmp-sensitivity">
        <div className="cmp-sensitivity__header">
          <span className="label-chip">Detection Sensitivity</span>
          <span className="mono cmp-sensitivity__val">Threshold: {diffThreshold}</span>
        </div>
        <input
          type="range"
          min="10"
          max="80"
          step="5"
          value={diffThreshold}
          onChange={(e) => setDiffThreshold(parseInt(e.target.value))}
          className="workspace__slider"
          aria-label="Difference threshold"
        />
        <div className="cmp-sensitivity__presets">
          <button className={`ws-preset-btn ${diffThreshold === 15 ? "active" : ""}`}
            onClick={() => setDiffThreshold(15)}>High (15) — Catches every small diff</button>
          <button className={`ws-preset-btn ${diffThreshold === 30 ? "active" : ""}`}
            onClick={() => setDiffThreshold(30)}>Balanced (30) — Recommended</button>
          <button className={`ws-preset-btn ${diffThreshold === 55 ? "active" : ""}`}
            onClick={() => setDiffThreshold(55)}>Strict (55) — Only major defects</button>
        </div>
        <p className="workspace__sensitivity-hint">
          {diffThreshold <= 20
            ? "High sensitivity: catches even faint color/lighting differences. May flag lighting/angle variations as defects."
            : diffThreshold >= 50
            ? "Strict: only flags large, obvious missing/added copper regions. Misses subtle defects."
            : "Balanced: best for real PCB photos taken from similar angles and lighting."}
        </p>
      </div>

      {/* ── Action buttons ── */}
      <div className="workspace__actions">
        <button
          className="ws-btn ws-btn--primary"
          onClick={handleCompare}
          disabled={!canCompare}
          aria-label="Compare PCBs"
        >
          {loading
            ? <><span className="ws-btn__spinner" aria-hidden="true" /> Comparing...</>
            : <><FiSearch size={16} /> Compare & Detect Defects</>
          }
        </button>
        {(refFile || testFile || result) && (
          <button className="ws-btn ws-btn--ghost" onClick={handleReset}>
            <FiRefreshCw size={16} /> Reset
          </button>
        )}
      </div>

      {/* ── Results area ── */}
      <div className="workspace__main">

        {loading && <AnalysisState preview={testPreview} />}

        {!loading && error && (
          <div className="workspace__error animate-fade-in">
            <FiAlertCircle size={28} className="workspace__error-icon" />
            <h3 className="workspace__error-title">Comparison Failed</h3>
            <p className="workspace__error-msg">{error}</p>
            <button className="ws-btn ws-btn--ghost" onClick={handleCompare}>Try Again</button>
          </div>
        )}

        {!loading && !error && result && (
          <div className="animate-fade-in">

            {/* Alignment status */}
            <div className={`cmp-align-badge ${result.aligned ? "cmp-align-badge--ok" : "cmp-align-badge--warn"}`}>
              {result.aligned
                ? <><FiCheckCircle size={14}/> Images aligned successfully using feature matching</>
                : <><FiAlertCircle size={14}/> Could not align images — results may include false positives from angle/zoom differences. Try uploading images taken from the same angle.</>
              }
            </div>

            {/* Viewer row */}
            <div className="workspace__viewer-row" style={{ marginTop: "var(--space-6)" }}>
              <div className="workspace__viewer-col">
                <div className="workspace__insp-id mono">
                  INSPECTION — {result.analysis.inspection_id}
                </div>
                {/* Show tabs: Original / Annotated / Diff Heatmap */}
                <CompareViewer
                  original={testPreview}
                  resultUrl={result.result_image_url}
                  diffUrl={result.diff_image_url}
                  detections={result.detections}
                  selectedIndex={selectedDet}
                  onSelectDetection={setSelectedDet}
                />
              </div>

              {result.detections.length > 0 && (
                <div className="workspace__det-col animate-fade-in">
                  <span className="label-chip" style={{ marginBottom: "var(--space-3)" }}>
                    Defect Regions ({result.total_detections})
                  </span>
                  <DetectionList
                    detections={result.detections}
                    selectedIndex={selectedDet}
                    onSelect={setSelectedDet}
                  />
                </div>
              )}
            </div>

            {/* Summary + Report */}
            <div className="workspace__results animate-fade-in">
              <div className="divider" />
              <span className="label-chip">Inspection Summary</span>
              <div style={{ marginTop: "var(--space-4)" }}>
                <InspectionSummary result={result} />
              </div>
              <div className="divider" />
              <div className="workspace__report-row">
                <div className="workspace__report-col">
                  <span className="label-chip" style={{ marginBottom: "var(--space-4)" }}>Report</span>
                  <ReportPanel
                    analysis={result.analysis}
                    detections={result.detections}
                    imageUrl={result.result_image_url}
                  />
                </div>
              </div>
              <div className="workspace__final-actions">
                <a href={result.result_image_url} download className="ws-btn ws-btn--ghost">
                  <FiDownload size={16}/> Download Annotated Image
                </a>
                <a href={result.diff_image_url} download className="ws-btn ws-btn--ghost">
                  <FiDownload size={16}/> Download Diff Heatmap
                </a>
                <button className="ws-btn ws-btn--ghost" onClick={handleReset}>
                  <FiRefreshCw size={16}/> New Inspection
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

// ── Inner viewer with 3 tabs: Original / Annotated / Diff Heatmap ─────────────
function CompareViewer({ original, resultUrl, diffUrl, detections, selectedIndex, onSelectDetection }) {
  const [tab, setTab] = useState("result");

  const src =
    tab === "diff"   ? diffUrl :
    tab === "result" ? resultUrl :
    original;

  return (
    <div className="pcb-viewer">
      <div className="pcb-viewer__toolbar">
        <div className="pcb-viewer__toggle">
          <button className={`pcb-viewer__tab ${tab === "original" ? "active" : ""}`}
            onClick={() => setTab("original")}>Original</button>
          <button className={`pcb-viewer__tab ${tab === "result" ? "active" : ""}`}
            onClick={() => setTab("result")} disabled={!resultUrl}>Annotated</button>
          <button className={`pcb-viewer__tab ${tab === "diff" ? "active" : ""}`}
            onClick={() => setTab("diff")} disabled={!diffUrl}>Diff Heatmap</button>
        </div>
      </div>
      <div className="pcb-viewer__canvas">
        <div className="pcb-viewer__img-wrap">
          {src
            ? <img src={src} alt={tab} className="pcb-viewer__img" draggable={false} />
            : <p style={{ color: "var(--text-muted)", padding: "2rem" }}>No image</p>
          }
        </div>
      </div>
    </div>
  );
}

export default CompareWorkspace;

