import { useEffect, useState } from "react";
import { FiImage, FiLayers } from "react-icons/fi";
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
        <div className="hero__glow" aria-hidden="true" />
        <div className="hero__inner">

          <div className="hero__content">
            <h1 className="hero__headline">
              <span>Precision PCB</span>
              <span><span className="hero__headline--accent">Defect</span> Inspection</span>
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
                <FiImage size={17} />
                <span>Single Image</span>
                <span className="hero__mode-badge">YOLO</span>
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === "hybrid"}
                className={`hero__mode-btn ${mode === "hybrid" ? "active" : ""}`}
                onClick={() => setMode("hybrid")}
              >
                <FiLayers size={17} />
                <span>Reference Compare</span>
                <span className="hero__mode-badge">HYBRID</span>
              </button>
            </div>
          </div>

          <div className="hero__visual" aria-hidden="true">
            <div className="hero__photo-frame">
              <img src="/hero-pcb.jpg" alt="" className="hero__photo" />
              <div className="hero__beam" />
              <div className="hero__bbox hero__bbox--white" />
              <div className="hero__bbox hero__bbox--red" />
              <div className="hero__defect-badge">
                <span className="hero__defect-icon">⚠</span> Defect Detected
              </div>
              <div className="hero__scanline" />
              <div className="hero__photo-corner hero__photo-corner--tl" />
              <div className="hero__photo-corner hero__photo-corner--tr" />
              <div className="hero__photo-corner hero__photo-corner--bl" />
              <div className="hero__photo-corner hero__photo-corner--br" />
            </div>
          </div>

        </div>
      </section>

      {mode === "single" ? <InspectionWorkspace /> : <HybridWorkspace />}
    </div>
  );
}

export default Home;
