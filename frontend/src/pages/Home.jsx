import { useEffect, useState } from "react";
import { FiCpu, FiLayers } from "react-icons/fi";
import Header from "../components/Header";
import InspectionWorkspace from "../components/InspectionWorkspace";
import HybridWorkspace from "../components/HybridWorkspace";
import api from "../services/api";
import "./Home.css";

function Home() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [mode, setMode] = useState("single");

  useEffect(() => {
    api.get("/health")
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <div className="home">
      <Header apiStatus={apiStatus} />

      <section className="hero">
        <div className="hero__inner">
          <div className="hero__content">
            <div className="label-chip" style={{ marginBottom: "var(--space-5)" }}>
              Computer Vision · YOLOv8 · PCB Quality Control
            </div>
            <h1 className="hero__headline">
              Precision PCB<br />
              <span className="hero__headline--accent">Defect Inspection</span>
            </h1>
            <p className="hero__sub">
              {mode === "single"
                ? "Upload a PCB image or capture one with your camera to automatically detect manufacturing defects using our trained YOLOv8 model."
                : "Upload a known-good reference board and a testing board. YOLOv8 finds defects, then reference alignment verifies whether the board actually changed."}
            </p>

            <div className="hero__mode-switcher" role="tablist" aria-label="Inspection mode">
              <button
                type="button"
                role="tab"
                aria-selected={mode === "single"}
                className={`hero__mode-btn ${mode === "single" ? "active" : ""}`}
                onClick={() => setMode("single")}
              >
                <FiCpu size={15} />
                Single Image
                <span className="hero__mode-badge">YOLO</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === "hybrid"}
                className={`hero__mode-btn ${mode === "hybrid" ? "active" : ""}`}
                onClick={() => setMode("hybrid")}
              >
                <FiLayers size={15} />
                Reference Compare
                <span className="hero__mode-badge">Hybrid</span>
              </button>
            </div>
          </div>
          <div className="hero__visual" aria-hidden="true">
            <PCBHeroGraphic />
          </div>
        </div>
      </section>

      {mode === "single" ? <InspectionWorkspace /> : <HybridWorkspace />}
    </div>
  );
}

function PCBHeroGraphic() {
  return (
    <svg
      className="hero-svg"
      viewBox="0 0 340 220"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Grid */}
      {[0,1,2,3,4,5,6].map(i => (
        <line key={`h${i}`} x1="10" y1={10+i*33} x2="330" y2={10+i*33}
          stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
      ))}
      {[0,1,2,3,4,5,6,7,8,9].map(i => (
        <line key={`v${i}`} x1={10+i*36} y1="10" x2={10+i*36} y2="210"
          stroke="rgba(255,255,255,0.04)" strokeWidth="1"/>
      ))}

      {/* PCB traces */}
      <path d="M46 43 L82 43 L82 76" stroke="rgba(101,211,145,0.25)" strokeWidth="1.5" fill="none"/>
      <path d="M82 76 L118 76 L118 109" stroke="rgba(101,211,145,0.25)" strokeWidth="1.5" fill="none"/>
      <path d="M190 43 L226 43 L226 76 L262 76" stroke="rgba(101,211,145,0.15)" strokeWidth="1.5" fill="none"/>
      <path d="M46 143 L82 143 L82 176 L154 176" stroke="rgba(215,255,79,0.15)" strokeWidth="1.5" fill="none"/>
      <path d="M262 109 L262 143 L226 143" stroke="rgba(101,211,145,0.15)" strokeWidth="1.5" fill="none"/>

      {/* Components (ICs) */}
      <rect x="54" y="30" width="36" height="24" rx="2" fill="rgba(23,28,28,0.9)" stroke="rgba(101,211,145,0.4)" strokeWidth="1.2"/>
      <rect x="62" y="36" width="20" height="12" rx="1" fill="rgba(101,211,145,0.08)"/>
      <rect x="64" y="39" width="16" height="6" rx="1" fill="rgba(101,211,145,0.2)"/>

      <rect x="180" y="30" width="52" height="36" rx="2" fill="rgba(23,28,28,0.9)" stroke="rgba(215,255,79,0.35)" strokeWidth="1.2"/>
      <rect x="190" y="40" width="32" height="16" rx="1" fill="rgba(215,255,79,0.06)"/>
      <rect x="194" y="44" width="24" height="8" rx="1" fill="rgba(215,255,79,0.15)"/>

      <rect x="108" y="96" width="28" height="28" rx="2" fill="rgba(23,28,28,0.9)" stroke="rgba(231,184,90,0.35)" strokeWidth="1.2"/>
      <rect x="116" y="104" width="12" height="12" rx="1" fill="rgba(231,184,90,0.15)"/>

      <rect x="216" y="96" width="36" height="24" rx="2" fill="rgba(23,28,28,0.9)" stroke="rgba(101,211,145,0.3)" strokeWidth="1.2"/>

      {/* Vias (dots) */}
      {[[82,76],[118,109],[226,76],[262,109],[154,176]].map(([cx,cy],i) => (
        <circle key={i} cx={cx} cy={cy} r="3.5" fill="rgba(255,255,255,0.06)" stroke="rgba(255,255,255,0.2)" strokeWidth="1"/>
      ))}
      {[[46,43],[190,43],[46,143],[262,143]].map(([cx,cy],i) => (
        <circle key={i+10} cx={cx} cy={cy} r="2.5" fill="rgba(101,211,145,0.3)"/>
      ))}

      {/* Detection box (animated) */}
      <rect x="100" y="88" width="48" height="44" rx="1"
        fill="rgba(255,98,92,0.05)" stroke="rgba(255,98,92,0.7)" strokeWidth="1.5"
        strokeDasharray="6 2"
        className="hero-detect-box"
      />
      <rect x="100" y="72" width="56" height="14" rx="1" fill="rgba(255,98,92,0.8)"/>
      <text x="104" y="83" fill="#000" fontFamily="IBM Plex Mono, monospace" fontSize="9" fontWeight="600">
        OPEN CIRCUIT 94%
      </text>

      {/* Scan line */}
      <line x1="10" y1="120" x2="330" y2="120"
        stroke="rgba(215,255,79,0.4)" strokeWidth="1"
        className="hero-scan"
      />

      {/* Coordinate overlay */}
      <text x="14" y="22" fill="rgba(255,255,255,0.12)" fontFamily="IBM Plex Mono, monospace" fontSize="8">640 × 640</text>
      <text x="268" y="22" fill="rgba(215,255,79,0.25)" fontFamily="IBM Plex Mono, monospace" fontSize="8">MODEL: YOLOv8</text>
    </svg>
  );
}

export default Home;
