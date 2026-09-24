import React, { useState, useEffect } from 'react';
import { Cpu, Award, BarChart3, Loader2, AlertCircle } from 'lucide-react';
import { fetchMetrics } from '../api';

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics()
      .then((res) => {
        setMetrics(res.data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="p-12 text-center text-slate-500 flex flex-col items-center justify-center gap-2">
        <Loader2 className="w-6 h-6 animate-spin text-emerald-800" />
        <span className="text-sm">Loading Model Performance Benchmarks...</span>
      </div>
    );
  }

  if (error || !metrics) {
    return (
      <div className="p-6 bg-rose-50 text-rose-800 rounded-lg border border-rose-200 flex items-center gap-3">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <span className="text-sm">{error || 'Failed to load model metrics.'}</span>
      </div>
    );
  }

  const { summary_metrics, model_comparison = [], per_scheme_f1 = [] } = metrics;

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl sm:text-2xl font-bold text-slate-900">Machine Learning Evaluation & Benchmarks</h2>
        <p className="text-xs sm:text-sm text-slate-500">
          Validation metrics and multi-model benchmark results trained on synthetic citizen profiles
        </p>
      </div>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-400">Macro F1 Score</span>
          <div className="text-2xl font-extrabold text-emerald-700 mt-1">
            {(summary_metrics.macro_f1 * 100).toFixed(1)}%
          </div>
          <span className="text-2xs text-slate-400">Balanced across rare schemes</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-400">Micro F1 Score</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">
            {(summary_metrics.micro_f1 * 100).toFixed(1)}%
          </div>
          <span className="text-2xs text-slate-400">Overall prediction accuracy</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-400">Exact Match Accuracy</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">
            {(summary_metrics.exact_match_accuracy * 100).toFixed(1)}%
          </div>
          <span className="text-2xs text-slate-400">All 20 schemes correct per row</span>
        </div>
        <div className="bg-white p-4 rounded-xl border border-slate-200 shadow-2xs">
          <span className="text-2xs font-bold uppercase text-slate-400">Model Architecture</span>
          <div className="text-lg font-bold text-emerald-800 mt-1">
            XGBoost (20 Trees)
          </div>
          <span className="text-2xs text-slate-400">MultiOutputClassifier</span>
        </div>
      </div>

      {/* Comparison Table */}
      {model_comparison.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs">
          <h3 className="text-base font-bold text-slate-900 mb-2">1. Multi-Model Benchmark Comparison</h3>
          <p className="text-xs text-slate-500 mb-4">
            Comparison of candidate models evaluated on the citizen validation dataset (80/20 train-test split):
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs sm:text-sm">
              <thead className="bg-slate-50 text-slate-700 font-bold border-b border-slate-200 uppercase text-2xs tracking-wider">
                <tr>
                  <th className="py-2.5 px-3">Model Architecture</th>
                  <th className="py-2.5 px-3">Exact Match</th>
                  <th className="py-2.5 px-3">F1 (Micro)</th>
                  <th className="py-2.5 px-3">F1 (Macro)</th>
                  <th className="py-2.5 px-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {model_comparison.map((m, idx) => {
                  const isBest = m.Model.toLowerCase().includes('xgboost');
                  return (
                    <tr key={idx} className={isBest ? 'bg-emerald-50/40 font-semibold' : 'hover:bg-slate-50'}>
                      <td className="py-2.5 px-3 flex items-center gap-2">
                        {m.Model}
                        {isBest && (
                          <span className="px-1.5 py-0.5 rounded text-3xs uppercase tracking-wider bg-emerald-800 text-white font-bold">
                            Selected
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono">{(m['Exact Match Accuracy'] * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono">{(m['F1 (micro)'] * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 font-mono">{(m['F1 (macro)'] * 100).toFixed(1)}%</td>
                      <td className="py-2.5 px-3 text-2xs text-slate-500">
                        {isBest ? 'Production Baseline' : 'Evaluated Candidate'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per Scheme F1 Breakdown */}
      {per_scheme_f1.length > 0 && (
        <div className="bg-white rounded-xl border border-slate-200 p-6 shadow-2xs">
          <h3 className="text-base font-bold text-slate-900 mb-2">2. Per-Scheme Test F1 Performance Breakdown</h3>
          <p className="text-xs text-slate-500 mb-4">
            Individual classification performance across all 20 evaluated welfare schemes:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
            {per_scheme_f1.map((item, idx) => (
              <div
                key={idx}
                className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-between"
              >
                <span className="font-semibold text-slate-800">{item.Scheme}</span>
                <span className="font-mono font-bold text-emerald-800">
                  F1: {(item.F1 * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
