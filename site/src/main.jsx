import React from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowUpRight,
  BarChart3,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  Database,
  Download,
  FileText,
  Github,
  Layers3,
  Microscope,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import "./styles.css";

const assets = {
  samples: "/assets/three-corpus-samples.webp",
  architecture: "/assets/aafnet-architecture.webp",
  performance: "/assets/three-corpus-performance.webp",
  coverage: "/assets/public-coverage.webp",
  parity: "/assets/parity-probes.webp",
  interpretability: "/assets/public-interpretability.webp",
  downstream: "/assets/downstream-inference.webp",
  gradcam: "/assets/gradcam-comparison.webp",
};

const releaseUrl =
  "https://github.com/crf0409/gjzw/releases/tag/three-corpus-release-20260619";
const repoUrl = "https://github.com/crf0409/gjzw";

const metrics = [
  ["Datasets", "3", "AL6, ASP-clean, AS25-clean"],
  ["Public CV", "60/60", "5 folds × 3 seeds × 4 model cells"],
  ["Follow-up probes", "180/180", "robustness, calibration and rotation"],
  ["Audit gate", "0 fail", "110 pass, 3 non-blocking warnings"],
];

const methods = [
  {
    icon: Layers3,
    title: "Architectural-aware fusion",
    text: "AAFNet combines multi-scale style attention with cross-scale gated fusion to preserve roof, facade and texture cues under viewpoint variation.",
  },
  {
    icon: BrainCircuit,
    title: "Contrastive distillation",
    text: "A supervised contrastive and focal-learning objective tightens class neighborhoods while retaining the ResNet-50 evidence trail used by the baselines.",
  },
  {
    icon: ShieldCheck,
    title: "Stress-tested evidence",
    text: "The release verifies robustness attribution, calibration, rotation behavior and public-corpus parity at fold level instead of relying on a single held-out split.",
  },
];

const evidence = [
  {
    kicker: "Figure 1",
    title: "Three-corpus visual scope",
    body: "Samples from the internal AL6 corpus and two public architectural-style datasets define the visual heterogeneity of the benchmark.",
    image: assets.samples,
    alt: "Three corpus sample grid for ancient building classification.",
  },
  {
    kicker: "Figure 2",
    title: "Cross-corpus performance",
    body: "The final audit requires completed fold-level evidence across ASP-clean and AS25-clean, aligned with the AL6 main-corpus analysis.",
    image: assets.performance,
    alt: "Three corpus performance figure.",
  },
  {
    kicker: "Figure 3",
    title: "Probe coverage matrix",
    body: "Every public-corpus cell is tracked through CV, robustness, calibration and rotation probes, with no missing terminal cells in the release manifest.",
    image: assets.coverage,
    alt: "Experiment coverage heatmap for public corpora.",
  },
];

const probeRows = [
  ["ASP-clean baseline", "15/15", "15/15", "15/15"],
  ["ASP-clean AAFNet", "15/15", "15/15", "15/15"],
  ["AS25-clean baseline", "15/15", "15/15", "15/15"],
  ["AS25-clean AAFNet", "15/15", "15/15", "15/15"],
];

const galleries = [
  {
    title: "Method Architecture",
    image: assets.architecture,
    alt: "AAFNet module architecture.",
  },
  {
    title: "Calibration and Rotation",
    image: assets.parity,
    alt: "Calibration and rotation parity probes.",
  },
  {
    title: "Public Interpretability",
    image: assets.interpretability,
    alt: "Public corpus interpretability panels.",
  },
  {
    title: "Downstream Inference",
    image: assets.downstream,
    alt: "Downstream inference effect cards.",
  },
  {
    title: "Grad-CAM Evidence",
    image: assets.gradcam,
    alt: "Grad-CAM comparison for model explanations.",
  },
];

function App() {
  return (
    <>
      <header className="site-header">
        <a className="brand" href="#top" aria-label="AAFNet home">
          <span className="brand-mark">A</span>
          <span>AAFNet</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#method">Method</a>
          <a href="#evidence">Evidence</a>
          <a href="#release">Release</a>
        </nav>
      </header>

      <main id="top">
        <section className="hero" aria-labelledby="hero-title">
          <img className="hero-bg" src={assets.samples} alt="" />
          <div className="hero-overlay" />
          <div className="hero-content">
            <p className="eyebrow">
              <Microscope size={17} aria-hidden="true" />
              Ancient-building image classification
            </p>
            <h1 id="hero-title">
              AAFNet: Architectural-Aware Multi-Scale Fusion
            </h1>
            <p className="hero-copy">
              A publication-ready project page for a three-corpus ancient-architecture
              benchmark, with fold-level evidence for performance, robustness,
              calibration, rotation behavior and interpretability.
            </p>
            <div className="hero-actions">
              <a className="button primary" href={releaseUrl}>
                <Download size={18} aria-hidden="true" />
                Release Package
              </a>
              <a className="button secondary" href={repoUrl}>
                <Github size={18} aria-hidden="true" />
                Source Code
              </a>
            </div>
          </div>
        </section>

        <section className="metrics-band" aria-label="Release metrics">
          {metrics.map(([label, value, note]) => (
            <article className="metric" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
              <p>{note}</p>
            </article>
          ))}
        </section>

        <section className="section intro-grid">
          <div>
            <p className="section-kicker">
              <BookOpen size={17} aria-hidden="true" />
              Project Thesis
            </p>
            <h2>Robust architectural recognition needs dataset parity.</h2>
          </div>
          <div className="intro-copy">
            <p>
              The project is organized around three corpora rather than a single
              favorable split. The release therefore records every public-corpus
              fold, seed and probe cell as source data, allowing the manuscript
              figures to be checked against the underlying experiment manifest.
            </p>
            <p>
              AAFNet is positioned as a compact architectural-aware enhancement
              over a ResNet-50 evidence trail, not as a detached black-box model.
              The website exposes the same audit logic used for the manuscript.
            </p>
          </div>
        </section>

        <section className="section method" id="method">
          <div className="section-heading">
            <p className="section-kicker">
              <Network size={17} aria-hidden="true" />
              Method
            </p>
            <h2>Architecture-aware evidence pipeline</h2>
          </div>
          <div className="method-layout">
            <figure className="figure-main">
              <img src={assets.architecture} alt="AAFNet architecture diagram." />
              <figcaption>
                AAFNet integrates multi-scale style attention, gated fusion and
                supervised contrastive distillation around a reproducible ResNet-50
                backbone.
              </figcaption>
            </figure>
            <div className="method-list">
              {methods.map(({ icon: Icon, title, text }) => (
                <article className="method-item" key={title}>
                  <Icon size={22} aria-hidden="true" />
                  <div>
                    <h3>{title}</h3>
                    <p>{text}</p>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="section evidence" id="evidence">
          <div className="section-heading wide">
            <p className="section-kicker">
              <BarChart3 size={17} aria-hidden="true" />
              Evidence
            </p>
            <h2>Figure-level data products are tied to completion checks.</h2>
          </div>
          <div className="evidence-list">
            {evidence.map((item) => (
              <article className="evidence-row" key={item.title}>
                <div className="evidence-text">
                  <span>{item.kicker}</span>
                  <h3>{item.title}</h3>
                  <p>{item.body}</p>
                </div>
                <figure>
                  <img src={item.image} alt={item.alt} loading="lazy" />
                </figure>
              </article>
            ))}
          </div>
        </section>

        <section className="section probe-section">
          <div className="section-heading">
            <p className="section-kicker">
              <CheckCircle2 size={17} aria-hidden="true" />
              Completion Matrix
            </p>
            <h2>All public follow-up probes are complete.</h2>
          </div>
          <div className="probe-table" role="table" aria-label="Public probe completion matrix">
            <div className="probe-head" role="row">
              <span role="columnheader">Cell</span>
              <span role="columnheader">Robustness</span>
              <span role="columnheader">Calibration</span>
              <span role="columnheader">Rotation</span>
            </div>
            {probeRows.map((row) => (
              <div className="probe-row" role="row" key={row[0]}>
                {row.map((cell, index) => (
                  <span role={index === 0 ? "rowheader" : "cell"} key={cell + index}>
                    {cell}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </section>

        <section className="gallery" aria-label="Research figure gallery">
          {galleries.map((item) => (
            <figure className="gallery-item" key={item.title}>
              <img src={item.image} alt={item.alt} loading="lazy" />
              <figcaption>{item.title}</figcaption>
            </figure>
          ))}
        </section>

        <section className="release" id="release">
          <div className="release-copy">
            <p className="section-kicker">
              <Database size={17} aria-hidden="true" />
              Reproducibility
            </p>
            <h2>Release package and source metadata</h2>
            <p>
              The GitHub release contains the full ZIP package, manifest and SHA256
              checksum. The repository stores the lightweight audit metadata and
              figure source tables under <code>experiment_data/three_corpus_release/</code>.
            </p>
          </div>
          <div className="release-actions">
            <a className="button primary" href={releaseUrl}>
              <FileText size={18} aria-hidden="true" />
              View Release
            </a>
            <a className="button secondary" href={repoUrl}>
              <ArrowUpRight size={18} aria-hidden="true" />
              Repository
            </a>
          </div>
          <p className="checksum">
            SHA256: d39a9f5683dc7348599104d3640533c73bfa59b6994eedccdae3889bd5d19748
          </p>
        </section>
      </main>

      <footer className="site-footer">
        <span>AAFNet research project</span>
        <span>
          <Sparkles size={15} aria-hidden="true" />
          Evidence-first, manuscript-ready presentation
        </span>
      </footer>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
