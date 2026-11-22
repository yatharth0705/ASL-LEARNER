import { Cpu, FileText, Info, Lock } from 'lucide-react';

export function AboutPage() {
  return (
    <div className="space-y-10 pb-16 max-w-4xl mx-auto">
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-accent mb-1">
          <Info className="size-3.5" />
          Technical Architecture & Attribution
        </div>
        <h1 className="text-3xl font-extrabold tracking-tight text-ink sm:text-4xl">
          About ASL-Learner
        </h1>
        <p className="mt-2 text-sm text-ink-muted leading-relaxed">
          An accessible, privacy-preserving American Sign Language learning application built with Google Antigravity.
        </p>
      </div>

      {/* Model Audit & Technical Decision */}
      <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8 space-y-5 shadow-xs">
        <div className="flex items-center gap-2 text-accent">
          <Cpu className="size-5" />
          <h2 className="text-lg font-bold text-ink">Pretrained Model Audit & Verification</h2>
        </div>

        <p className="text-xs text-ink-muted leading-relaxed">
          Per project requirements, <strong>no model training, fine-tuning, or re-training</strong> was conducted. 
          The application exclusively utilizes verified pretrained model weights obtained directly from official open-source GitHub repositories:
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border border-line rounded-xl overflow-hidden">
            <thead className="bg-surface-muted text-ink-muted border-b border-line">
              <tr>
                <th className="p-3 font-semibold">Repository</th>
                <th className="p-3 font-semibold">Model Artifact</th>
                <th className="p-3 font-semibold">Architecture</th>
                <th className="p-3 font-semibold">Size</th>
                <th className="p-3 font-semibold">Latency (i5 CPU)</th>
                <th className="p-3 font-semibold">License</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              <tr className="bg-accent-soft/20 font-medium">
                <td className="p-3 text-ink font-semibold">GestureSign (Selected)</td>
                <td className="p-3 font-mono text-[11px]">model.onnx</td>
                <td className="p-3">Conv-Transformer (1.4M)</td>
                <td className="p-3">5.62 MB</td>
                <td className="p-3 text-emerald-600 font-bold">~4.4 ms</td>
                <td className="p-3">MIT License</td>
              </tr>
              <tr className="text-ink-muted">
                <td className="p-3">ASL-translator (Evaluated)</td>
                <td className="p-3 font-mono text-[11px]">sign_transformer_best.pth</td>
                <td className="p-3">Transformer Encoder (2.4M)</td>
                <td className="p-3">9.85 MB</td>
                <td className="p-3">~5.8 ms</td>
                <td className="p-3">MIT License</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="space-y-2 pt-2 text-xs text-ink-muted leading-relaxed">
          <h3 className="font-bold text-ink">Why GestureSign Was Selected:</h3>
          <ul className="list-disc pl-5 space-y-1">
            <li>
              <strong>Browser-Native WebAssembly:</strong> Optimized for on-device inference via ONNX Runtime Web without requiring a local Python desktop process or server GPU.
            </li>
            <li>
              <strong>Aspect-Ratio Invariance:</strong> Compensates for landscape laptop webcams stretching vertical proportions relative to mobile training footage.
            </li>
            <li>
              <strong>Automatic Handedness:</strong> Automatically canonicalizes signing hand into the dominant slot, making it equally accurate for left- and right-handed signers.
            </li>
            <li>
              <strong>Metric Embedding Space:</strong> Produces a 128-dimensional embedding vector used for cosine-similarity practice scoring.
            </li>
            <li>
              <strong>Calibrated Confidence:</strong> Employs temperature scaling (0.8827) and a strict 0.2903 threshold to report "Not sure" instead of guessing randomly when hands are resting.
            </li>
          </ul>
        </div>
      </section>

      {/* Privacy Guarantee */}
      <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8 space-y-4 shadow-xs">
        <div className="flex items-center gap-2 text-emerald-600">
          <Lock className="size-5" />
          <h2 className="text-lg font-bold text-ink">Privacy & Security Commitment</h2>
        </div>

        <p className="text-xs text-ink-muted leading-relaxed">
          Your video stream is processed <strong>strictly inside your browser memory</strong> using MediaPipe Holistic and Web Workers. 
          No camera frames, facial coordinates, or sign recordings are ever transmitted across the network, stored on a remote server, or shared with third parties.
        </p>
      </section>

      {/* Open Source Licenses & Legal Attribution */}
      <section className="rounded-2xl border border-line bg-surface p-6 sm:p-8 space-y-4 shadow-xs">
        <div className="flex items-center gap-2 text-ink">
          <FileText className="size-5" />
          <h2 className="text-lg font-bold text-ink">Open Source Attributions & Licenses</h2>
        </div>

        <div className="space-y-3 text-xs text-ink-muted leading-relaxed">
          <div className="rounded-xl border border-line bg-paper p-4">
            <h4 className="font-semibold text-ink">GestureSign</h4>
            <p className="mt-1">
              Source: <a href="https://github.com/shashank11yadav/GestureSign" target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">github.com/shashank11yadav/GestureSign</a>
            </p>
            <p className="mt-1 font-mono text-[11px]">
              Copyright (c) 2026 Shashank Yadav. Licensed under the MIT License.
            </p>
          </div>

          <div className="rounded-xl border border-line bg-paper p-4">
            <h4 className="font-semibold text-ink">ASL-translator</h4>
            <p className="mt-1">
              Source: <a href="https://github.com/aymenwnia/ASL-translator" target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">github.com/aymenwnia/ASL-translator</a>
            </p>
            <p className="mt-1 font-mono text-[11px]">
              Copyright (c) 2026 AWAINIA Aymen. Licensed under the MIT License.
            </p>
          </div>

          <div className="rounded-xl border border-line bg-paper p-4">
            <h4 className="font-semibold text-ink">Google ASL Signs Dataset</h4>
            <p className="mt-1">
              Collected by Deaf Professional Arts Network (DPAN) for the Kaggle "Google - American Sign Language Signs" competition.
            </p>
            <p className="mt-1 font-mono text-[11px]">
              Licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
