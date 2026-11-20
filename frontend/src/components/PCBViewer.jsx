import { useState, useRef } from "react";
import { FiZoomIn, FiZoomOut, FiMaximize2, FiMinimize2 } from "react-icons/fi";
import "./PCBViewer.css";

const DEFECT_COLORS = [
  "#2EEE92", "#55EE9E", "#F5B544", "#FA1F21",
  "#7DD3FC", "#C4B5FD", "#FCA5A5", "#86EFAC",
];

function PCBViewer({ original, resultUrl, detections, selectedIndex, onSelectDetection }) {
  const [userView, setUserView]   = useState(null);
  const [zoom, setZoom]           = useState(1);
  const [fullscreen, setFs]       = useState(false);
  const imgRef                    = useRef(null);
  const [imgDims, setImgDims]     = useState({ w: 1, h: 1 });
  const [prevResult, setPrevResult] = useState(resultUrl);

  // Derived: auto-switch to detection when result arrives; respect user override
  if (resultUrl !== prevResult) {
    setPrevResult(resultUrl);
    if (resultUrl) setUserView("result");
    else setUserView(null);
  }

  const view    = userView ?? (resultUrl ? "result" : "original");
  const setView = (v) => setUserView(v);

  const onImgLoad = (e) => {
    setImgDims({ w: e.target.naturalWidth, h: e.target.naturalHeight });
  };

  const zoomIn  = () => setZoom((z) => Math.min(z + 0.25, 4));
  const zoomOut = () => setZoom((z) => Math.max(z - 0.25, 0.5));
  const reset   = () => setZoom(1);

  const src = view === "result" && resultUrl ? resultUrl : original;

  if (!original) {
    return (
      <div className="pcb-viewer pcb-viewer--empty">
        <div className="pcb-viewer__empty-icon">
          <PCBGridSVG />
        </div>
        <p className="pcb-viewer__empty-text">No image loaded</p>
        <p className="pcb-viewer__empty-sub">Upload or capture a PCB to begin</p>
      </div>
    );
  }

  const content = (
    <div className={`pcb-viewer${fullscreen ? " pcb-viewer--fullscreen" : ""}`}>
      <div className="pcb-viewer__toolbar">
        <div className="pcb-viewer__toggle">
          <button
            className={`pcb-viewer__tab ${view === "original" ? "active" : ""}`}
            onClick={() => setView("original")}
          >
            Original
          </button>
          <button
            className={`pcb-viewer__tab ${view === "result" ? "active" : ""}`}
            onClick={() => setView("result")}
            disabled={!resultUrl}
          >
            Detection
          </button>
        </div>
        <div className="pcb-viewer__controls">
          <button className="pcb-viewer__btn" onClick={zoomOut} aria-label="Zoom out"><FiZoomOut size={15} /></button>
          <span className="pcb-viewer__zoom mono">{Math.round(zoom * 100)}%</span>
          <button className="pcb-viewer__btn" onClick={zoomIn}  aria-label="Zoom in"><FiZoomIn  size={15} /></button>
          <button className="pcb-viewer__btn" onClick={reset}   aria-label="Reset zoom">1:1</button>
          <button
            className="pcb-viewer__btn"
            onClick={() => setFs(!fullscreen)}
            aria-label={fullscreen ? "Exit fullscreen" : "Fullscreen"}
          >
            {fullscreen ? <FiMinimize2 size={15} /> : <FiMaximize2 size={15} />}
          </button>
        </div>
      </div>

      <div className="pcb-viewer__canvas">
        <div
          className="pcb-viewer__img-wrap"
          style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
        >
          <img
            ref={imgRef}
            src={src}
            alt={view === "original" ? "Original PCB" : "Detection result"}
            className="pcb-viewer__img"
            onLoad={onImgLoad}
            draggable={false}
          />
          {view === "original" && detections && detections.length > 0 && (
            <svg
              className="pcb-viewer__overlay"
              viewBox={`0 0 ${imgDims.w} ${imgDims.h}`}
              preserveAspectRatio="none"
            >
              {detections.map((det, i) => {
                const [x1, y1, x2, y2] = det.bbox;
                const w = x2 - x1;
                const h = y2 - y1;
                const color = DEFECT_COLORS[i % DEFECT_COLORS.length];
                const isSelected = selectedIndex === i;
                return (
                  <g
                    key={i}
                    onClick={() => onSelectDetection(isSelected ? null : i)}
                    className="pcb-viewer__bbox-group"
                    role="button"
                    aria-label={`${det.type} — ${(det.confidence * 100).toFixed(1)}%`}
                  >
                    <rect
                      x={x1} y={y1} width={w} height={h}
                      fill={isSelected ? `${color}22` : "none"}
                      stroke={color}
                      strokeWidth={isSelected ? 2 : 1.2}
                      className={`pcb-viewer__bbox${isSelected ? " pcb-viewer__bbox--selected" : ""}`}
                    />
                    <rect
                      x={x1} y={y1 - 20}
                      width={Math.max(w, 60)} height={18}
                      fill={color}
                      rx={1}
                    />
                    <text
                      x={x1 + 4} y={y1 - 6}
                      fill="#000"
                      fontSize={Math.max(12, imgDims.w * 0.012)}
                      fontFamily="IBM Plex Mono, monospace"
                      fontWeight="500"
                    >
                      {det.type.replace(/_/g," ").toUpperCase()} {(det.confidence * 100).toFixed(0)}%
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </div>
      </div>

      {detections && detections.length > 0 && view === "original" && (
        <div className="pcb-viewer__legend">
          <span className="pcb-viewer__legend-label mono">
            {detections.length} detection{detections.length !== 1 ? "s" : ""} · click to select
          </span>
        </div>
      )}
    </div>
  );

  if (fullscreen) {
    return (
      <div className="pcb-viewer__fs-backdrop" onClick={() => setFs(false)}>
        <div onClick={(e) => e.stopPropagation()} style={{ width: "100%", maxWidth: 1100 }}>
          {content}
        </div>
      </div>
    );
  }

  return content;
}

function PCBGridSVG() {
  return (
    <svg width="120" height="80" viewBox="0 0 120 80" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="10" y1="10" x2="110" y2="10" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="10" y1="30" x2="110" y2="30" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="10" y1="50" x2="110" y2="50" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="10" y1="70" x2="110" y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="10" y1="10" x2="10"  y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="35" y1="10" x2="35"  y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="60" y1="10" x2="60"  y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="85" y1="10" x2="85"  y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <line x1="110" y1="10" x2="110" y2="70" stroke="rgba(255,255,255,0.06)" strokeWidth="1"/>
      <rect x="25" y="20" width="20" height="12" rx="1" fill="none" stroke="rgba(46,238,146,0.3)" strokeWidth="1"/>
      <rect x="55" y="38" width="28" height="16" rx="1" fill="none" stroke="rgba(46,238,146,0.3)" strokeWidth="1"/>
      <line x1="35" y1="26" x2="55" y2="46" stroke="rgba(46,238,146,0.2)" strokeWidth="1"/>
      <circle cx="10"  cy="10" r="2.5" fill="rgba(255,255,255,0.12)"/>
      <circle cx="110" cy="10" r="2.5" fill="rgba(255,255,255,0.12)"/>
      <circle cx="10"  cy="70" r="2.5" fill="rgba(255,255,255,0.12)"/>
      <circle cx="110" cy="70" r="2.5" fill="rgba(255,255,255,0.12)"/>
    </svg>
  );
}

export default PCBViewer;
