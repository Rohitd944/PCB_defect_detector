import "./Header.css";

function Header({ apiStatus }) {
  const statusText = apiStatus === "online" ? "Online"
    : apiStatus === "offline" ? "Offline" : "Checking";

  return (
    <header className="cs-header">
      <div className="cs-header__inner">

        <div className="cs-header__brand">
          <div className="cs-header__logo">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.6" />
              <rect x="8" y="8" width="8" height="8" rx="1" fill="currentColor" opacity="0.25" />
              <path d="M7 1v2M12 1v2M17 1v2M7 21v2M12 21v2M17 21v2M1 7h2M1 12h2M1 17h2M21 7h2M21 12h2M21 17h2" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
            </svg>
          </div>
          <div className="cs-header__brand-text">
            <div className="cs-header__title">PCB_DEFECT_DETECTION</div>
            <div className="cs-header__sub">AI Vision Inspection System</div>
          </div>
        </div>

        <div className="cs-header__meta">
          <div className="cs-header__chip">
            <span className="cs-header__dot" />
            <span className="cs-header__chip-label">Model:</span>
            <span className="cs-header__chip-value">YOLOv8</span>
          </div>
          <div className="cs-header__chip">
            <span className={`cs-header__dot ${apiStatus === "offline" ? "offline" : ""}`} />
            <span className="cs-header__chip-label">Status:</span>
            <span className="cs-header__chip-value">{statusText}</span>
          </div>
        </div>

      </div>
    </header>
  );
}

export default Header;
