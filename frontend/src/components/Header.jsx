import React from 'react';
import { Landmark, ShieldCheck, Cpu, Award } from 'lucide-react';

export default function Header({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'check', label: 'Eligibility Check' },
    { id: 'schemes', label: 'Scheme Directory' },
    { id: 'metrics', label: 'Model Performance' },
    { id: 'about', label: 'About Platform' },
  ];

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-11 h-11 rounded-lg bg-emerald-800 text-white flex items-center justify-center shadow-sm flex-shrink-0">
              <Landmark className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl sm:text-2xl font-bold text-slate-900 tracking-tight">
                  TN Scheme Eligibility Predictor
                </h1>
                <span className="hidden sm:inline-block px-2 py-0.5 text-xs font-semibold bg-emerald-50 text-emerald-800 rounded border border-emerald-200">
                  Govt of Tamil Nadu
                </span>
              </div>
              <p className="text-xs sm:text-sm text-slate-500">
                Check eligibility for 20 Tamil Nadu welfare schemes using AI/ML decision intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-slate-100 p-1 rounded-lg border border-slate-200">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-3 py-1.5 text-xs sm:text-sm font-semibold rounded-md transition-all ${
                    activeTab === tab.id
                      ? 'bg-white text-slate-900 shadow-sm'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
