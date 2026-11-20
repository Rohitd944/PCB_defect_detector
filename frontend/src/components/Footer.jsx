import "./Footer.css";

function Footer() {
  return (
    <footer className="cs-footer">
      <div className="cs-footer__inner">
        <div className="cs-footer__brand">
          <span className="cs-footer__dot" />
          <span>PCB_DEFECT_DETECTION</span>
        </div>
        <p className="cs-footer__stack mono">
          YOLOv8 · React · FastAPI · OpenCV
        </p>
        <p className="cs-footer__copy mono">
          © 2026
        </p>
      </div>
    </footer>
  );
}

export default Footer;
