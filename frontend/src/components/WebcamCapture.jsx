import { useRef, useState } from "react";
import Webcam from "react-webcam";
import { FiCamera, FiX, FiRefreshCw } from "react-icons/fi";
import "./WebcamCapture.css";

function WebcamCapture({ onCapture, onClose, title = "Camera Input" }) {
  const webcamRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [captured, setCaptured] = useState(null);
  const [dimensions, setDimensions] = useState("—");

  const handleUserMedia = (stream) => {
    setReady(true);
    const track = stream.getVideoTracks()[0];
    if (track) {
      const settings = track.getSettings();
      if (settings.width && settings.height) {
        setDimensions(`${settings.width} × ${settings.height}`);
      }
    }
  };

  const captureImage = () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) setCaptured(imageSrc);
  };

  const retake = () => setCaptured(null);

  const confirm = () => {
    if (captured) {
      onCapture(captured);
    }
  };

  return (
    <div className="webcam-overlay animate-fade-in" role="dialog" aria-label="Camera capture">
      <div className="webcam-panel">

        <div className="webcam-header">
          <div className="webcam-header__left">
            <span className="label-chip">
              <FiCamera size={11} />
              {title}
            </span>
          </div>
          <button className="webcam-close" onClick={onClose} aria-label="Close camera">
            <FiX size={18} />
          </button>
        </div>

        <div className="webcam-viewport">
          {!captured ? (
            <>
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                className="webcam-feed"
                onUserMedia={handleUserMedia}
                onUserMediaError={() => setReady(false)}
                videoConstraints={{ facingMode: "environment" }}
              />
              <div className="webcam-guides">
                <div className="webcam-corner webcam-corner--tl" />
                <div className="webcam-corner webcam-corner--tr" />
                <div className="webcam-corner webcam-corner--bl" />
                <div className="webcam-corner webcam-corner--br" />
              </div>
              <div className="webcam-overlay-text">
                <span className="mono">{ready ? "CAMERA ACTIVE" : "INITIALIZING..."}</span>
                <span className="mono">FRAME {dimensions}</span>
                <span className={`mono webcam-status-badge ${ready ? "ready" : "wait"}`}>
                  {ready ? "READY" : "WAIT"}
                </span>
              </div>
            </>
          ) : (
            <img src={captured} alt="Captured PCB" className="webcam-captured" />
          )}
        </div>

        <div className="webcam-controls">
          {!captured ? (
            <button
              className="btn-capture"
              onClick={captureImage}
              disabled={!ready}
              aria-label="Capture image"
            >
              <span className="btn-capture__ring" />
              <FiCamera size={20} />
            </button>
          ) : (
            <div className="webcam-confirm-row">
              <button className="btn-ghost" onClick={retake}>
                <FiRefreshCw size={16} />
                Retake
              </button>
              <button className="btn-primary" onClick={confirm}>
                <FiCamera size={16} />
                Use This Image
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}

export default WebcamCapture;
