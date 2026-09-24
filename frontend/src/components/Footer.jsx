import React from 'react';

export default function Footer() {
  return (
    <footer className="mt-16 border-t border-slate-200 bg-white py-8 text-xs text-slate-500">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <span className="font-bold text-slate-800">TN Scheme Eligibility Predictor</span> • AI/ML Civic-Tech Decision Platform
          <p className="text-2xs text-slate-400 mt-0.5">
            Designed for PBL Project Review (CS5403 Machine Learning)
          </p>
        </div>
        <div className="flex items-center gap-4 text-slate-600">
          <a
            href="https://www.tnesevai.tn.gov.in"
            target="_blank"
            rel="noreferrer"
            className="hover:text-emerald-800 transition underline underline-offset-2"
          >
            TN e-Sevai Portal
          </a>
          <a
            href="https://kmut.tn.gov.in"
            target="_blank"
            rel="noreferrer"
            className="hover:text-emerald-800 transition underline underline-offset-2"
          >
            KMUT Portal
          </a>
          <a
            href="https://pudhumaipenn.tn.gov.in"
            target="_blank"
            rel="noreferrer"
            className="hover:text-emerald-800 transition underline underline-offset-2"
          >
            Pudhumai Penn
          </a>
        </div>
      </div>
    </footer>
  );
}
